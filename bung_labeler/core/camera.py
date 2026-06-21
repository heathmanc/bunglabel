from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

import cv2

try:
    from pypylon import pylon
except ImportError:
    pylon = None


@dataclass
class CameraOpenResult:
    ok: bool
    message: str
    backend_name: str = ""
    width: float = 0
    height: float = 0
    fps: float = 0


def _source_to_device(source: str | int) -> str | None:
    if isinstance(source, int):
        return f"/dev/video{source}"
    if isinstance(source, str) and source.startswith("/dev/video"):
        return source
    return None


def _fourcc_to_str(value: float) -> str:
    try:
        v = int(value or 0)
        return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip()
    except Exception:
        return ""


def force_v4l2_format(source: str | int, width: int | None, height: int | None, fps: int | None, pixelformat: str = "MJPG") -> str:
    """Use v4l2-ctl to force an exact USB camera mode before OpenCV opens it.

    This is useful on Jetson/Linux because OpenCV sometimes silently falls back to
    a slower YUYV mode even when CAP_PROP_FOURCC is requested.
    """
    device = _source_to_device(source)
    if not device:
        return "v4l2-ctl force skipped: source is not a /dev/video device."
    if shutil.which("v4l2-ctl") is None:
        return "v4l2-ctl force skipped: v4l2-ctl not installed."

    cmd = ["v4l2-ctl", "-d", device]
    if width and height:
        cmd.append(f"--set-fmt-video=width={width},height={height},pixelformat={pixelformat}")
    if fps:
        cmd.append(f"--set-parm={fps}")

    if len(cmd) <= 3:
        return "v4l2-ctl force skipped: width/height/fps not specified."

    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=3)
        if proc.returncode == 0:
            return "v4l2-ctl forced mode: " + " ".join(cmd)
        return f"v4l2-ctl force failed: {proc.stderr.strip() or proc.stdout.strip()}"
    except Exception as e:
        return f"v4l2-ctl force exception: {e}"


class CameraSource:
    """Camera/video wrapper with OpenCV and Basler/Pylon support.

    Basler/Pylon returns normal OpenCV-compatible BGR numpy frames so the
    existing labeling, capture, adjustment, and export code can stay unchanged.
    """

    BACKENDS = {
        "Auto": cv2.CAP_ANY,
        "V4L2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
        "GStreamer": getattr(cv2, "CAP_GSTREAMER", cv2.CAP_ANY),
        "FFmpeg": getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY),
    }

    def __init__(self) -> None:
        self.cap = None
        self.converter = None
        self.source: str | int | None = None
        self.last_result = CameraOpenResult(False, "Not opened")

        self.threaded = True
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_ok = False
        self._read_fps = 0.0
        self._frame_counter = 0
        self._fps_t0 = time.perf_counter()

    def _set_basler_value(self, node_name: str, value) -> bool:
        if self.cap is None or pylon is None:
            return False
        try:
            node = getattr(self.cap, node_name, None)
            if node is None:
                return False
            if hasattr(node, "IsWritable") and not node.IsWritable():
                return False
            node.SetValue(value)
            return True
        except Exception:
            return False

    def _get_basler_value(self, node_name: str, default=0):
        if self.cap is None or pylon is None:
            return default
        try:
            node = getattr(self.cap, node_name, None)
            if node is None:
                return default
            return node.GetValue()
        except Exception:
            return default

    def _basler_node_limit(self, node_name: str, attr: str, default: int = 0) -> int:
        try:
            node = getattr(self.cap, node_name, None)
            fn = getattr(node, attr, None)
            if callable(fn):
                return int(fn())
        except Exception:
            pass
        return int(default)

    def _set_basler_int_value(self, node_name: str, value: int | None, *, use_max_if_none: bool = False) -> bool:
        if self.cap is None or pylon is None:
            return False
        try:
            node = getattr(self.cap, node_name, None)
            if node is None:
                return False
            if hasattr(node, "IsWritable") and not node.IsWritable():
                return False
            min_v = self._basler_node_limit(node_name, "GetMin", 0)
            max_v = self._basler_node_limit(node_name, "GetMax", int(value or 0))
            inc = max(1, self._basler_node_limit(node_name, "GetInc", 1))
            if value is None or int(value) <= 0:
                target = max_v if use_max_if_none else min_v
            else:
                target = max(min_v, min(max_v, int(value)))
            if inc > 1:
                target = min_v + ((target - min_v) // inc) * inc
            node.SetValue(int(target))
            return True
        except Exception:
            return False

    def _set_basler_aoi(self, width: int | None, height: int | None) -> str:
        """Set Basler AOI deterministically.

        Basler Width/Height are sensor AOI controls, not display scaling. If a
        smaller AOI was previously selected, simply omitting Width/Height can
        leave the camera stuck at that old size. Reset offsets to their minimum
        and explicitly set Width/Height on every open so the requested main
        resolution is what appears in the live preview.
        """
        messages = []
        # Offsets can constrain the maximum accepted Width/Height. Reset them
        # first so growing back to full sensor or another large AOI works.
        self._set_basler_int_value("OffsetX", None, use_max_if_none=False)
        self._set_basler_int_value("OffsetY", None, use_max_if_none=False)

        width_ok = self._set_basler_int_value("Width", width, use_max_if_none=True)
        height_ok = self._set_basler_int_value("Height", height, use_max_if_none=True)
        actual_w = int(float(self._get_basler_value("Width", 0) or 0))
        actual_h = int(float(self._get_basler_value("Height", 0) or 0))
        if width_ok or height_ok:
            messages.append(f"AOI {actual_w}x{actual_h}")
        else:
            messages.append("AOI unchanged")
        return ", ".join(messages)


    def _set_basler_exposure(self, auto: bool, exposure_us: int | None) -> str:
        """Apply Basler exposure settings. Returns a short operator-readable status."""
        if self.cap is None or pylon is None:
            return "Basler exposure skipped: camera is not open."
        messages = []
        if auto:
            if self._set_basler_value("ExposureAuto", "Continuous"):
                messages.append("auto exposure on")
            elif self._set_basler_value("ExposureAuto", "Once"):
                messages.append("auto exposure once")
            else:
                messages.append("auto exposure not writable")
        else:
            self._set_basler_value("ExposureAuto", "Off")
            value = int(exposure_us or 0)
            if value > 0:
                if self._set_basler_value("ExposureTime", float(value)):
                    messages.append(f"manual exposure {value} us")
                elif self._set_basler_value("ExposureTimeAbs", float(value)):
                    messages.append(f"manual exposure {value} us")
                else:
                    messages.append("manual exposure not writable")
            else:
                messages.append("manual exposure selected, no value set")
        return ", ".join(messages)

    def set_exposure(self, auto: bool = True, exposure_us: int | None = None) -> str:
        """Apply exposure settings to the currently opened camera.

        Basler/Pylon uses microseconds. OpenCV/V4L2 exposure units vary by driver,
        so for non-Basler cameras the numeric value is passed through directly.
        """
        if self.cap is None:
            return "Exposure skipped: camera is not open."
        try:
            if self.last_result.backend_name == "Basler/Pylon":
                return self._set_basler_exposure(auto, exposure_us)
            if auto:
                ok = bool(self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75))
                return "auto exposure on" if ok else "auto exposure may not be supported by this backend"
            # V4L2 commonly uses 0.25 for manual. Some cameras ignore it.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            value = int(exposure_us or 0)
            ok = bool(self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))) if value else False
            return f"manual exposure set to {value}" if ok else "manual exposure may not be supported by this backend"
        except Exception as e:
            return f"Exposure apply failed: {e}"

    def _open_basler(self, source: str | int, width: int | None, height: int | None, fps: int | None, threaded: bool, exposure_auto: bool = True, exposure_us: int | None = None) -> bool:
        if pylon is None:
            self.last_result = CameraOpenResult(False, "Basler/Pylon selected, but pypylon is not installed in this Python environment. Run: pip install pypylon")
            return False

        tl = pylon.TlFactory.GetInstance()
        devices = tl.EnumerateDevices()
        if not devices:
            self.last_result = CameraOpenResult(False, "No Basler cameras detected through Pylon. Check Pylon Viewer, USB cable/port, and udev permissions.")
            return False

        # Source is optional for Basler. If source text matches a serial number,
        # use that camera; otherwise use the first Pylon-detected camera.
        source_text = str(source).strip() if source is not None else ""
        device = devices[0]
        if source_text and not source_text.isdigit() and not source_text.startswith("/dev/"):
            for d in devices:
                try:
                    if source_text in (d.GetSerialNumber(), d.GetModelName(), d.GetFriendlyName()):
                        device = d
                        break
                except Exception:
                    pass

        try:
            self.cap = pylon.InstantCamera(tl.CreateDevice(device))
            self.cap.Open()

            aoi_msg = self._set_basler_aoi(width, height)
            if fps:
                # Some Basler models use AcquisitionFrameRateEnable before AcquisitionFrameRate.
                self._set_basler_value("AcquisitionFrameRateEnable", True)
                self._set_basler_value("AcquisitionFrameRate", float(fps))

            exposure_msg = self._set_basler_exposure(exposure_auto, exposure_us)

            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self.cap.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception as e:
            self.last_result = CameraOpenResult(False, f"Basler/Pylon open exception: {e}")
            self.close()
            return False

        ok_any = False
        last_shape = None
        try:
            for _ in range(5):
                ok, frame = self._read_basler_frame(timeout_ms=5000)
                if ok and frame is not None:
                    ok_any = True
                    last_shape = frame.shape
                    with self._lock:
                        self._latest_frame = frame
                        self._latest_ok = True
                    break
                time.sleep(0.05)
        except Exception as e:
            self.last_result = CameraOpenResult(False, f"Basler/Pylon opened, but frame grab failed: {e}")
            self.close()
            return False

        actual_w = float(self._get_basler_value("Width", 0) or 0)
        actual_h = float(self._get_basler_value("Height", 0) or 0)
        actual_fps = float(self._get_basler_value("ResultingFrameRate", self._get_basler_value("AcquisitionFrameRate", 0)) or 0)
        try:
            model = device.GetModelName()
            serial = device.GetSerialNumber()
        except Exception:
            model, serial = "Basler", "unknown"

        if not ok_any:
            self.last_result = CameraOpenResult(False, "Basler/Pylon camera opened, but no frames were readable.", "Basler/Pylon", actual_w, actual_h, actual_fps)
            self.close()
            return False

        self.last_result = CameraOpenResult(
            True,
            f"Opened Basler/Pylon camera {model} serial {serial}. Actual size {actual_w:.0f}x{actual_h:.0f}, reported FPS {actual_fps:.1f}. {aoi_msg}. Exposure: {exposure_msg}. First frame shape: {last_shape}.",
            "Basler/Pylon",
            actual_w,
            actual_h,
            actual_fps,
        )

        if threaded:
            self._running = True
            self._fps_t0 = time.perf_counter()
            self._frame_counter = 0
            self._read_fps = 0.0
            self._thread = threading.Thread(target=self._reader_loop, name="BungVisionCameraReader", daemon=True)
            self._thread.start()

        return True

    def _read_basler_frame(self, timeout_ms: int = 5000):
        if self.cap is None or pylon is None:
            return False, None
        if not self.cap.IsGrabbing():
            return False, None
        grab = self.cap.RetrieveResult(timeout_ms, pylon.TimeoutHandling_Return)
        try:
            if grab is None or not grab.GrabSucceeded():
                return False, None
            frame = self.converter.Convert(grab).GetArray() if self.converter is not None else grab.Array
            return True, frame
        finally:
            if grab is not None:
                grab.Release()

    def open(
        self,
        source: str | int,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        backend: str = "Auto",
        warmup_frames: int = 3,
        low_latency: bool = True,
        mjpg: bool = True,
        threaded: bool = True,
        force_v4l2: bool = False,
        exposure_auto: bool = True,
        exposure_us: int | None = None,
    ) -> bool:
        self.close()
        self.source = source
        self.threaded = threaded
        force_message = ""

        if backend == "Basler/Pylon":
            return self._open_basler(source, width, height, fps, threaded=threaded, exposure_auto=exposure_auto, exposure_us=exposure_us)

        if force_v4l2:
            force_message = force_v4l2_format(source, width, height, fps, pixelformat="MJPG")

        backend_id = self.BACKENDS.get(backend, cv2.CAP_ANY)

        try:
            if backend_id == cv2.CAP_ANY:
                self.cap = cv2.VideoCapture(source)
            else:
                self.cap = cv2.VideoCapture(source, backend_id)
        except Exception as e:
            self.last_result = CameraOpenResult(False, f"OpenCV VideoCapture exception: {e}")
            self.cap = None
            return False

        if self.cap is None or not self.cap.isOpened():
            self.last_result = CameraOpenResult(False, f"Could not open source {source!r} using backend {backend}.")
            self.close()
            return False

        if low_latency:
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        # Order matters for some V4L2 cameras/OpenCV builds. Set dimensions,
        # request MJPG, then set dimensions/FPS again so the final negotiated
        # mode is more likely to be MJPG at the desired size.
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if mjpg:
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        exposure_msg = self.set_exposure(exposure_auto, exposure_us)

        ok_any = False
        last_shape = None
        for _ in range(max(1, warmup_frames)):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                ok_any = True
                last_shape = frame.shape
                with self._lock:
                    self._latest_frame = frame
                    self._latest_ok = True
                break
            time.sleep(0.05)

        backend_name = ""
        try:
            backend_name = self.cap.getBackendName()
        except Exception:
            backend_name = backend

        actual_w = float(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_h = float(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
        actual_fourcc = _fourcc_to_str(self.cap.get(cv2.CAP_PROP_FOURCC) or 0)

        if not ok_any:
            self.last_result = CameraOpenResult(
                False,
                f"Source opened with {backend_name}, but no frames were readable. Try another source index, lower resolution, or V4L2 backend.",
                backend_name,
                actual_w,
                actual_h,
                actual_fps,
            )
            self.close()
            return False

        self.last_result = CameraOpenResult(
            True,
            f"Opened source {source!r} using {backend_name}. Actual size {actual_w:.0f}x{actual_h:.0f}, FOURCC {actual_fourcc or 'unknown'}, reported FPS {actual_fps:.1f}. Exposure: {exposure_msg}. First frame shape: {last_shape}. {force_message}",
            backend_name,
            actual_w,
            actual_h,
            actual_fps,
        )

        if threaded:
            self._running = True
            self._fps_t0 = time.perf_counter()
            self._frame_counter = 0
            self._read_fps = 0.0
            self._thread = threading.Thread(target=self._reader_loop, name="BungVisionCameraReader", daemon=True)
            self._thread.start()

        return True

    def _reader_loop(self) -> None:
        while self._running and self.is_open():
            if self.last_result.backend_name == "Basler/Pylon":
                ok, frame = self._read_basler_frame(timeout_ms=1000)
            else:
                ok, frame = self.cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._latest_ok = True

                self._frame_counter += 1
                now = time.perf_counter()
                elapsed = now - self._fps_t0
                if elapsed >= 1.0:
                    self._read_fps = self._frame_counter / elapsed
                    self._frame_counter = 0
                    self._fps_t0 = now
            else:
                with self._lock:
                    self._latest_ok = False
                time.sleep(0.005)

    def read(self):
        if not self.is_open():
            return False, None

        if self.threaded:
            with self._lock:
                if self._latest_frame is None:
                    return False, None
                return self._latest_ok, self._latest_frame.copy()

        if self.last_result.backend_name == "Basler/Pylon":
            return self._read_basler_frame(timeout_ms=1000)

        ok, frame = self.cap.read()
        if ok and frame is not None:
            return True, frame
        return False, None

    def read_fps(self) -> float:
        return float(self._read_fps)

    def drain(self, count: int = 2) -> None:
        # In threaded mode, the reader loop already keeps only the newest frame.
        if self.threaded:
            return
        if self.cap is None or not self.cap.isOpened():
            return
        for _ in range(max(0, count)):
            try:
                self.cap.grab()
            except Exception:
                return

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            try:
                self._thread.join(timeout=0.5)
            except Exception:
                pass
        self._thread = None

        if self.cap is not None:
            try:
                if self.last_result.backend_name == "Basler/Pylon" and pylon is not None:
                    if self.cap.IsGrabbing():
                        self.cap.StopGrabbing()
                    if self.cap.IsOpen():
                        self.cap.Close()
                else:
                    self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.converter = None
        with self._lock:
            self._latest_frame = None
            self._latest_ok = False

    def is_open(self) -> bool:
        if self.cap is None:
            return False
        try:
            if self.last_result.backend_name == "Basler/Pylon" and pylon is not None:
                return bool(self.cap.IsOpen() and self.cap.IsGrabbing())
            return bool(self.cap.isOpened())
        except Exception:
            return False


def quick_test_source(source: str | int, backend: str = "Auto", width: int | None = None, height: int | None = None, exposure_auto: bool = True, exposure_us: int | None = None) -> CameraOpenResult:
    cam = CameraSource()
    try:
        cam.open(source, width=width, height=height, backend=backend, warmup_frames=5, low_latency=True, mjpg=True, threaded=False, force_v4l2=(backend != "Basler/Pylon"), exposure_auto=exposure_auto, exposure_us=exposure_us)
        return cam.last_result
    finally:
        cam.close()

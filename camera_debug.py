#!/usr/bin/env python3
"""
Standalone camera debug tool for BungVision.

Run outside Docker from the BungVision folder:
  source .venv/bin/activate
  python camera_debug.py --source 0 --backend v4l2

Try:
  python camera_debug.py --source 0 --backend auto
  python camera_debug.py --source 1 --backend v4l2
  python camera_debug.py --source /dev/video0 --backend v4l2
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time

import cv2


def parse_source(s: str):
    try:
        return int(s)
    except ValueError:
        return s


def backend_id(name: str):
    name = name.lower()
    if name == "v4l2":
        return cv2.CAP_V4L2
    if name == "gstreamer":
        return cv2.CAP_GSTREAMER
    if name == "ffmpeg":
        return cv2.CAP_FFMPEG
    return cv2.CAP_ANY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--backend", default="v4l2", choices=["auto", "v4l2", "gstreamer", "ffmpeg"])
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--fps", type=int, default=0)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--no-mjpg", action="store_true", help="Do not request MJPG camera format")
    ap.add_argument("--buffer", type=int, default=1, help="Requested camera buffer size")
    ap.add_argument("--force-v4l2", action="store_true", help="Run v4l2-ctl to force MJPG mode before OpenCV opens")
    args = ap.parse_args()

    src = parse_source(args.source)
    bid = backend_id(args.backend)

    if args.force_v4l2 and shutil.which("v4l2-ctl"):
        device = f"/dev/video{src}" if isinstance(src, int) else str(src)
        cmd = ["v4l2-ctl", "-d", device]
        if args.width and args.height:
            cmd.append(f"--set-fmt-video=width={args.width},height={args.height},pixelformat=MJPG")
        if args.fps:
            cmd.append(f"--set-parm={args.fps}")
        print("Forcing V4L2 mode:", " ".join(cmd))
        subprocess.run(cmd, check=False)

    print(f"Opening source={src!r}, backend={args.backend}")
    cap = cv2.VideoCapture(src, bid) if bid != cv2.CAP_ANY else cv2.VideoCapture(src)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, args.buffer)
    except Exception:
        pass
    if not args.no_mjpg:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print("FAILED: camera did not open")
        return 1

    try:
        print("Backend:", cap.getBackendName())
    except Exception:
        pass

    print("Actual width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Actual height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Actual fps:", cap.get(cv2.CAP_PROP_FPS))
    fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    fourcc = "".join([chr((fourcc_val >> 8 * i) & 0xFF) for i in range(4)])
    print("Actual FOURCC:", fourcc)
    print("Reading frames. Press q to quit window, or Ctrl+C.")

    count = 0
    last = time.perf_counter()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Frame read failed")
            time.sleep(0.2)
            continue

        count += 1
        now = time.perf_counter()
        if now - last >= 1.0:
            fps = count / (now - last)
            count = 0
            last = now
            print(f"OK frame shape={frame.shape}, fps={fps:.1f}")

        if not args.no_window:
            cv2.putText(frame, f"BungVision camera debug FPS {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("BungVision Camera Debug", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    cap.release()
    if not args.no_window:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

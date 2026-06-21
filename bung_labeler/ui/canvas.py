from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
from PySide6.QtCore import QPoint, QPointF, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap, QPolygonF, QWheelEvent
from PySide6.QtWidgets import QWidget


@dataclass
class Box:
    """Canvas annotation.

    kind="box" stores x/y/w/h like regular YOLO detect labels.
    kind="obb" stores four image-space corner points in clockwise order:
    top-left, top-right, bottom-right, bottom-left.
    """

    x: float
    y: float
    w: float
    h: float
    class_id: int = 0
    label: str = "bung"
    kind: str = "box"
    points: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.kind = str(self.kind or "box").lower()
        if self.kind == "obb" and not self.points:
            self.points = [
                [float(self.x), float(self.y)],
                [float(self.x + self.w), float(self.y)],
                [float(self.x + self.w), float(self.y + self.h)],
                [float(self.x), float(self.y + self.h)],
            ]
        self._refresh_bounds()

    def _refresh_bounds(self) -> None:
        if self.kind == "obb" and self.points:
            xs = [float(p[0]) for p in self.points]
            ys = [float(p[1]) for p in self.points]
            self.x = min(xs)
            self.y = min(ys)
            self.w = max(xs) - self.x
            self.h = max(ys) - self.y

    def to_dict(self) -> dict:
        self._refresh_bounds()
        data = {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "class_id": self.class_id,
            "label": self.label,
            "kind": self.kind,
        }
        if self.kind == "obb":
            data["points"] = [[float(x), float(y)] for x, y in self.points]
        return data

    @staticmethod
    def from_dict(d: dict) -> "Box":
        kind = str(d.get("kind") or d.get("type") or "box").lower()
        pts = d.get("points") or d.get("obb") or []
        if kind == "obb" and pts and len(pts) >= 4:
            pts = [[float(p[0]), float(p[1])] for p in pts[:4]]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return Box(
                min(xs),
                min(ys),
                max(xs) - min(xs),
                max(ys) - min(ys),
                int(d.get("class_id", 0)),
                d.get("label", "battery"),
                "obb",
                pts,
            )
        return Box(float(d["x"]), float(d["y"]), float(d["w"]), float(d["h"]), int(d.get("class_id", 0)), d.get("label", "bung"), "box")


class ImageCanvas(QWidget):
    boxes_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.pixmap: QPixmap | None = None
        self.image_path: Path | None = None
        self.image_w = 0
        self.image_h = 0
        self.boxes: list[Box] = []
        self.drawing = False
        self.start_pos = QPoint()
        self.current_pos = QPoint()
        self.selected_idx: int | None = None
        self.selected_handle_idx: int | None = None
        self.dragging_handle = False
        self.class_name = "battery"
        self.class_id = 0
        self.annotation_kind = "box"
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.panning = False
        self.pan_start = QPoint()
        self.pan_origin = QPoint()
        # Saved annotation visibility and preview-only model-test overlays are separate layers.
        # Model-test overlays are never saved as annotations and are not baked into the pixmap.
        self.show_annotations: bool = True
        self.model_test_overlays: list[dict] = []

        # v0.9.36 canvas-performance cache. The old canvas recomputed a smooth
        # scaled QPixmap inside _target_rect(), which was called many times per
        # paint/mouse-move. With 5 MP Basler images this made panning slower as
        # soon as annotations existed. Cache the display pixmap for the current
        # widget size + zoom and reuse it while panning.
        self._fit_size = QSize()
        self._scaled_pixmap: QPixmap | None = None
        self._scaled_cache_key: tuple[int, int, float, int] | None = None
    def set_annotation_kind(self, kind: str) -> None:
        self.annotation_kind = str(kind or "box").lower()
        self.update()

    def _invalidate_display_cache(self) -> None:
        self._fit_size = QSize()
        self._scaled_pixmap = None
        self._scaled_cache_key = None

    def resizeEvent(self, event) -> None:
        self._invalidate_display_cache()
        super().resizeEvent(event)

    def _base_fit_size(self) -> QSize:
        if not self.pixmap or self.image_w <= 0 or self.image_h <= 0 or self.width() <= 0 or self.height() <= 0:
            return QSize()
        if self._fit_size.isValid() and self._fit_size.width() > 0 and self._fit_size.height() > 0:
            return self._fit_size
        img_w = max(1, self.image_w)
        img_h = max(1, self.image_h)
        scale = min(self.width() / img_w, self.height() / img_h)
        scale = max(0.001, scale)
        self._fit_size = QSize(max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        return self._fit_size

    def _display_pixmap(self, target_size: QSize) -> QPixmap | None:
        if not self.pixmap or not target_size.isValid() or target_size.width() <= 0 or target_size.height() <= 0:
            return None
        source_key = int(self.pixmap.cacheKey())
        key = (int(target_size.width()), int(target_size.height()), round(float(self.zoom), 4), source_key)
        if self._scaled_pixmap is not None and self._scaled_cache_key == key:
            return self._scaled_pixmap
        # Smooth scaling is expensive, so do it only when image/zoom/viewport size changes.
        self._scaled_pixmap = self.pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_cache_key = key
        return self._scaled_pixmap

    def load_image(self, path: Path) -> bool:
        frame = cv2.imread(str(path))
        if frame is None:
            return False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.pixmap = QPixmap.fromImage(qimg)
        self._invalidate_display_cache()
        self.image_path = path
        self.image_w = w
        self.image_h = h
        self.boxes = []
        self.model_test_overlays = []
        self.selected_idx = None
        self.selected_handle_idx = None
        self.fit_to_window()
        self.update()
        return True

    def set_frame(self, frame_bgr) -> None:
        if frame_bgr is None:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.pixmap = QPixmap.fromImage(qimg)
        self._invalidate_display_cache()
        self.image_w = w
        self.image_h = h
        self.update()

    def set_boxes_from_dicts(self, boxes: list[dict]) -> None:
        self.boxes = [Box.from_dict(b) for b in boxes]
        self.selected_idx = None
        self.selected_handle_idx = None
        self.update()
        self.boxes_changed.emit()

    def clear_boxes(self) -> None:
        self.boxes = []
        self.selected_idx = None
        self.selected_handle_idx = None
        self.update()
        self.boxes_changed.emit()


    def set_model_test_overlays(self, items: list[dict]) -> None:
        """Set temporary preview overlays from model testing. These are not labels."""
        self.model_test_overlays = list(items or [])
        self.update()

    def clear_model_test_overlays(self) -> None:
        """Clear only temporary model-test preview overlays."""
        self.model_test_overlays = []
        self.update()

    def set_annotation_visibility(self, visible: bool) -> None:
        """Show/hide saved labels without deleting them."""
        self.show_annotations = bool(visible)
        if not self.show_annotations:
            self.selected_idx = None
            self.selected_handle_idx = None
        self.update()

    def clear_all_visual_overlays(self) -> None:
        """Clear preview overlays and hide saved labels without modifying label data."""
        self.model_test_overlays = []
        self.set_annotation_visibility(False)

    def delete_selected(self) -> None:
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.boxes):
            del self.boxes[self.selected_idx]
            self.selected_idx = None
            self.selected_handle_idx = None
            self.update()
            self.boxes_changed.emit()

    def nudge_selected(self, dx: float, dy: float) -> None:
        if self.selected_idx is None or not (0 <= self.selected_idx < len(self.boxes)):
            return
        b = self.boxes[self.selected_idx]
        if b.kind == "obb" and self.selected_handle_idx is not None:
            b.points[self.selected_handle_idx][0] = max(0, min(self.image_w, b.points[self.selected_handle_idx][0] + dx))
            b.points[self.selected_handle_idx][1] = max(0, min(self.image_h, b.points[self.selected_handle_idx][1] + dy))
            b._refresh_bounds()
        elif b.kind == "obb":
            for p in b.points:
                p[0] = max(0, min(self.image_w, p[0] + dx))
                p[1] = max(0, min(self.image_h, p[1] + dy))
            b._refresh_bounds()
        else:
            b.x = max(0, min(self.image_w - b.w, b.x + dx))
            b.y = max(0, min(self.image_h - b.h, b.y + dy))
        self.update()
        self.boxes_changed.emit()

    def keyPressEvent(self, event) -> None:
        step = 10 if event.modifiers() & Qt.ShiftModifier else 1
        if event.key() == Qt.Key_Left:
            self.nudge_selected(-step, 0)
        elif event.key() == Qt.Key_Right:
            self.nudge_selected(step, 0)
        elif event.key() == Qt.Key_Up:
            self.nudge_selected(0, -step)
        elif event.key() == Qt.Key_Down:
            self.nudge_selected(0, step)
        else:
            super().keyPressEvent(event)

    def fit_to_window(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update()

    def zoom_in(self) -> None:
        self.set_zoom(self.zoom * 1.25)

    def zoom_out(self) -> None:
        self.set_zoom(self.zoom / 1.25)

    def set_zoom(self, value: float, anchor: QPoint | None = None) -> None:
        old_rect = self._target_rect() if self.pixmap else QRect()
        old_zoom = self.zoom
        self.zoom = max(0.25, min(8.0, float(value)))
        if old_zoom != self.zoom:
            self._scaled_pixmap = None
            self._scaled_cache_key = None
        if anchor is not None and self.pixmap and old_rect.width() > 0 and old_rect.height() > 0 and old_zoom != self.zoom:
            rel_x = (anchor.x() - old_rect.x()) / old_rect.width()
            rel_y = (anchor.y() - old_rect.y()) / old_rect.height()
            new_rect = self._target_rect(base_pan=False)
            self.pan_x = int(anchor.x() - (new_rect.x() + rel_x * new_rect.width()))
            self.pan_y = int(anchor.y() - (new_rect.y() + rel_y * new_rect.height()))
        self.update()

    def _target_rect(self, base_pan: bool = True) -> QRect:
        if not self.pixmap:
            return QRect()
        base = self._base_fit_size()
        if not base.isValid() or base.width() <= 0 or base.height() <= 0:
            return QRect()
        w = max(1, int(base.width() * self.zoom))
        h = max(1, int(base.height() * self.zoom))
        x = (self.width() - w) // 2
        y = (self.height() - h) // 2
        if base_pan:
            x += self.pan_x
            y += self.pan_y
        return QRect(x, y, w, h)

    def _screen_to_image(self, p: QPoint) -> tuple[float, float] | None:
        r = self._target_rect()
        if not r.contains(p) or self.image_w == 0 or self.image_h == 0:
            return None
        ix = (p.x() - r.x()) / r.width() * self.image_w
        iy = (p.y() - r.y()) / r.height() * self.image_h
        return max(0, min(self.image_w, ix)), max(0, min(self.image_h, iy))

    def _image_to_screen_point(self, x: float, y: float) -> QPointF:
        r = self._target_rect()
        sx = r.x() + float(x) / max(1, self.image_w) * r.width()
        sy = r.y() + float(y) / max(1, self.image_h) * r.height()
        return QPointF(sx, sy)

    def _image_to_screen_rect(self, b: Box) -> QRect:
        r = self._target_rect()
        sx = r.x() + int(b.x / max(1, self.image_w) * r.width())
        sy = r.y() + int(b.y / max(1, self.image_h) * r.height())
        sw = int(b.w / max(1, self.image_w) * r.width())
        sh = int(b.h / max(1, self.image_h) * r.height())
        return QRect(sx, sy, sw, sh).normalized()

    def _screen_polygon(self, b: Box) -> QPolygonF:
        if b.kind == "obb" and b.points:
            return QPolygonF([self._image_to_screen_point(x, y) for x, y in b.points])
        br = self._image_to_screen_rect(b)
        return QPolygonF([QPointF(br.left(), br.top()), QPointF(br.right(), br.top()), QPointF(br.right(), br.bottom()), QPointF(br.left(), br.bottom())])

    def _handle_at_screen_pos(self, pos: QPoint) -> tuple[int, int] | None:
        handle_radius = 10
        for i, b in enumerate(self.boxes):
            if b.kind != "obb" or not b.points:
                continue
            for j, (x, y) in enumerate(b.points):
                sp = self._image_to_screen_point(x, y)
                if abs(sp.x() - pos.x()) <= handle_radius and abs(sp.y() - pos.y()) <= handle_radius:
                    return i, j
        return None

    def _box_at_screen_pos(self, pos: QPoint) -> int | None:
        hits: list[tuple[float, int]] = []
        for i, b in enumerate(self.boxes):
            if b.kind == "obb":
                poly = self._screen_polygon(b)
                if poly.containsPoint(QPointF(pos), Qt.OddEvenFill):
                    hits.append((max(1.0, b.w * b.h), i))
            elif self._image_to_screen_rect(b).contains(pos):
                hits.append((max(1.0, b.w * b.h), i))
        if not hits:
            return None
        hits.sort(key=lambda item: item[0])
        return hits[0][1]

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.pixmap is None:
            return
        steps = event.angleDelta().y() / 120.0
        if steps > 0:
            self.set_zoom(self.zoom * (1.20 ** steps), event.position().toPoint())
        elif steps < 0:
            self.set_zoom(self.zoom / (1.20 ** abs(steps)), event.position().toPoint())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.pixmap is None:
            return

        self.setFocus()
        pos = event.position().toPoint()
        mods = event.modifiers()
        alt_down = bool(mods & Qt.AltModifier)
        ctrl_down = bool(mods & Qt.ControlModifier)

        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and alt_down):
            self.panning = True
            self.pan_start = pos
            self.pan_origin = QPoint(self.pan_x, self.pan_y)
            self.setCursor(Qt.ClosedHandCursor)
            return

        handle = self._handle_at_screen_pos(pos)
        if event.button() == Qt.LeftButton and handle is not None:
            self.selected_idx, self.selected_handle_idx = handle
            self.dragging_handle = True
            self.setCursor(Qt.SizeAllCursor)
            self.update()
            return

        if event.button() == Qt.RightButton or (event.button() == Qt.LeftButton and ctrl_down):
            self.selected_idx = self._box_at_screen_pos(pos)
            self.selected_handle_idx = None
            self.update()
            return

        if event.button() != Qt.LeftButton:
            return

        if self._screen_to_image(pos) is not None:
            self.drawing = True
            self.start_pos = pos
            self.current_pos = pos
            self.selected_idx = None
            self.selected_handle_idx = None
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position().toPoint()
        if self.panning:
            delta = pos - self.pan_start
            self.pan_x = self.pan_origin.x() + delta.x()
            self.pan_y = self.pan_origin.y() + delta.y()
            self.update()
            return
        if self.dragging_handle and self.selected_idx is not None and self.selected_handle_idx is not None:
            img_pos = self._screen_to_image(pos)
            if img_pos is not None:
                b = self.boxes[self.selected_idx]
                b.points[self.selected_handle_idx] = [float(img_pos[0]), float(img_pos[1])]
                b._refresh_bounds()
                # Do not emit boxes_changed while the mouse is moving. That signal
                # can autosave/recount/repaint side panels every few milliseconds.
                # Emit once on release instead.
                self.update()
            return
        if self.drawing:
            self.current_pos = pos
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.panning:
            self.panning = False
            self.unsetCursor()
            self.update()
            return
        if self.dragging_handle:
            self.dragging_handle = False
            self.unsetCursor()
            self.boxes_changed.emit()
            return
        if not self.drawing:
            return
        self.drawing = False
        release_pos = event.position().toPoint()
        p1 = self._screen_to_image(self.start_pos)
        p2 = self._screen_to_image(release_pos)
        if p1 is None or p2 is None:
            self.update()
            return
        x1, y1 = p1
        x2, y2 = p2
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)

        if w < 8 or h < 8:
            self.selected_idx = self._box_at_screen_pos(release_pos)
            self.update()
            return

        if self.annotation_kind == "obb":
            pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            self.boxes.append(Box(x, y, w, h, int(self.class_id), self.class_name, "obb", pts))
        else:
            self.boxes.append(Box(x, y, w, h, int(self.class_id), self.class_name, "box"))
        self.selected_idx = len(self.boxes) - 1
        self.selected_handle_idx = None
        self.boxes_changed.emit()
        self.update()

    def _class_color(self, b: Box, selected: bool = False) -> QColor:
        if selected:
            return QColor(250, 204, 21)
        label = str(b.label or "").lower()
        if label.startswith("battery") or b.class_id == 0:
            return QColor(96, 165, 250)
        if label.startswith("bung") or b.class_id == 1:
            return QColor(34, 197, 94)
        if label.startswith("retainer") or b.class_id == 2:
            return QColor(168, 85, 247)
        return QColor(251, 146, 60)


    def _draw_model_test_overlays(self, p: QPainter) -> None:
        if not self.model_test_overlays:
            return
        for item in self.model_test_overlays:
            typ = str(item.get("type", "")).lower()
            if typ in ("battery_obb", "battery_box", "bung_obb", "bung_box"):
                is_battery = typ.startswith("battery")
                status = str(item.get("status", "raw")).lower()
                if status == "outside":
                    color = QColor(250, 204, 21)
                elif status == "ignored":
                    color = QColor(148, 163, 184)
                elif is_battery:
                    color = QColor(96, 165, 250)
                else:
                    color = QColor(34, 197, 94)

                br = QRect()
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(color, 3 if is_battery else 2))
                if typ.endswith("_obb"):
                    pts = item.get("points", [])
                    if len(pts) >= 4:
                        poly = QPolygonF([self._image_to_screen_point(float(x), float(y)) for x, y in pts[:4]])
                        p.drawPolygon(poly)
                        xs = [float(x) for x, _y in pts[:4]]
                        ys = [float(y) for _x, y in pts[:4]]
                        b = Box(min(xs), min(ys), max(1, max(xs) - min(xs)), max(1, max(ys) - min(ys)), 0, "model", "box")
                        br = self._image_to_screen_rect(b)
                else:
                    x1, y1, x2, y2 = [float(v) for v in item.get("xyxy", [0, 0, 0, 0])]
                    b = Box(x1, y1, max(1, x2 - x1), max(1, y2 - y1), 0, "model", "box")
                    br = self._image_to_screen_rect(b)
                    p.drawRect(br)

                cx = float(item.get("cx", 0)); cy = float(item.get("cy", 0))
                if cx == 0 and cy == 0 and not br.isNull():
                    cx = (float(item.get("xyxy", [0, 0, 0, 0])[0]) + float(item.get("xyxy", [0, 0, 0, 0])[2])) / 2.0
                    cy = (float(item.get("xyxy", [0, 0, 0, 0])[1]) + float(item.get("xyxy", [0, 0, 0, 0])[3])) / 2.0
                csp = self._image_to_screen_point(cx, cy)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(color, 2))
                p.drawEllipse(csp, 4, 4)

                label = str(item.get("label", "battery" if is_battery else "bung"))
                if is_battery:
                    sp = self._image_to_screen_point(cx, cy)
                    p.fillRect(QRect(int(sp.x() - 125), int(sp.y() - 30), 250, 22), QColor(0, 0, 0, 175))
                    p.setPen(QPen(color, 1))
                    p.drawText(int(sp.x() - 120), int(sp.y() - 14), label)
                elif not br.isNull():
                    p.fillRect(QRect(br.x(), max(0, br.y() - 20), max(80, min(180, len(label) * 8 + 12)), 20), QColor(0, 0, 0, 160))
                    p.setPen(QPen(color, 1))
                    p.drawText(br.x() + 5, max(14, br.y() - 5), label)
            elif typ == "pattern_point":
                status = str(item.get("status", "expected")).lower()
                if status == "found":
                    color = QColor(34, 197, 94)
                elif status == "missing":
                    color = QColor(239, 68, 68)
                elif status == "extra":
                    color = QColor(250, 204, 21)
                else:
                    color = QColor(14, 165, 233)
                x = float(item.get("x", 0)); y = float(item.get("y", 0))
                sp = self._image_to_screen_point(x, y)
                tol_px = float(item.get("tol_px", 0))
                if tol_px > 0 and self.image_w > 0:
                    r = self._target_rect()
                    tol_screen = max(5, int(tol_px / max(1, self.image_w) * r.width()))
                    p.setPen(QPen(color, 1, Qt.DashLine))
                    p.setBrush(Qt.NoBrush)
                    p.drawEllipse(sp, tol_screen, tol_screen)
                p.setPen(QPen(color, 3))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(sp, 7, 7)
                label = str(item.get("label", "point"))
                text = label if status == "expected" else f"{label} {status.upper()}"
                p.fillRect(QRect(int(sp.x() + 9), int(sp.y() - 13), max(80, min(220, len(text) * 8 + 12)), 22), QColor(0, 0, 0, 170))
                p.setPen(QPen(color, 1))
                p.drawText(int(sp.x() + 14), int(sp.y() + 3), text)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(17, 24, 39))
        if self.pixmap:
            r = self._target_rect()
            display_pm = self._display_pixmap(r.size())
            if display_pm is not None:
                p.drawPixmap(r.x(), r.y(), display_pm)
            else:
                p.drawPixmap(r, self.pixmap)
            low_detail = bool(self.panning)
            p.setPen(QPen(QColor(55, 65, 81), 2))
            p.drawRect(r)
            p.fillRect(QRect(10, 10, 185, 24), QColor(0, 0, 0, 150))
            p.setPen(QColor(203, 213, 225))
            p.drawText(18, 28, f"Zoom {self.zoom:.2f}x | Tool {self.annotation_kind.upper()}")

            if self.show_annotations:
                for i, b in enumerate(self.boxes):
                    selected = i == self.selected_idx
                    color = self._class_color(b, selected)
                    p.setPen(QPen(color, 3 if selected else 2))
                    p.setBrush(Qt.NoBrush)
                    if b.kind == "obb":
                        poly = self._screen_polygon(b)
                        p.drawPolygon(poly)
                        br = self._image_to_screen_rect(b)
                        if not low_detail:
                            # Draw corner handles for every OBB; make selected ones larger/brighter.
                            for j, (x, y) in enumerate(b.points):
                                sp = self._image_to_screen_point(x, y)
                                size = 12 if selected else 8
                                handle_rect = QRect(int(sp.x() - size / 2), int(sp.y() - size / 2), size, size)
                                p.setBrush(Qt.NoBrush)
                                p.setPen(QPen(QColor(250, 204, 21) if selected and j == self.selected_handle_idx else color, 2))
                                p.drawRect(handle_rect)
                                p.drawText(int(sp.x() + 6), int(sp.y() - 6), str(j + 1))
                    else:
                        br = self._image_to_screen_rect(b)
                        p.drawRect(br)

                    if not low_detail:
                        tag = f"{b.label} [{b.kind.upper()}]"
                        tag_w = max(90, min(170, len(tag) * 8 + 14))
                        p.fillRect(QRect(br.x(), max(br.y() - 22, 0), tag_w, 22), QColor(0, 0, 0, 170))
                        p.setPen(QPen(color, 1))
                        p.drawText(br.x() + 5, max(br.y() - 6, 14), tag)

            if not low_detail:
                self._draw_model_test_overlays(p)

            if self.drawing:
                draw_color = QColor(96, 165, 250) if self.class_name == "battery" or self.class_id == 0 else QColor(34, 197, 94)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(draw_color, 2, Qt.DashLine))
                p.drawRect(QRect(self.start_pos, self.current_pos).normalized())
        else:
            p.setPen(QColor(156, 163, 175))
            p.drawText(self.rect(), Qt.AlignCenter, "Open a camera or load/capture an image")

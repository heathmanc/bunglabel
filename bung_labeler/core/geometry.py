"""Pure geometry helpers for BungVision Label Studio.

No Qt or OpenCV dependency, so these can be unit tested headlessly. The
point-in-polygon test here is the single canonical implementation used by both
the review/quantity logic (``core/review.py``) and the model count-test overlay.
"""
from __future__ import annotations

import math


def point_in_polygon(x: float, y: float, poly) -> bool:
    """Ray-casting point-in-polygon test. Accepts any sequence of (x, y) points."""
    pts = [(float(p[0]), float(p[1])) for p in poly]
    n = len(pts)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def rect_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two axis-aligned rectangles given as (x, y, w, h)."""
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix = max(0.0, min(ax2, bx2) - max(ax, bx))
    iy = max(0.0, min(ay2, by2) - max(ay, by))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def normalize_angle_deg(angle: float) -> float:
    """Normalize an image-space angle to [-90, 90) degrees for readable skew."""
    angle = float(angle)
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def polygon_long_edge_angle(pts) -> tuple[float | None, float]:
    """Return the angle of the longest edge and its length from a 4-point OBB polygon.

    Ultralytics xywhr angles can appear wrong for long rectangular parts because
    the model/formatter may swap width/height or use a different angle convention.
    For a battery, the useful plant-floor angle is usually the long-edge skew, so
    compute it directly from the drawn polygon.
    """
    if pts is None or len(pts) < 4:
        return None, 0.0
    best_angle = None
    best_len = 0.0
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        length = math.hypot(dx, dy)
        if length > best_len:
            best_len = length
            best_angle = normalize_angle_deg(math.degrees(math.atan2(dy, dx)))
    return best_angle, best_len

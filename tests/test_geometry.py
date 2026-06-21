"""Unit tests for the pure geometry helpers (headless-safe)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bung_labeler.core import geometry as g

SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]


def test_point_inside_and_outside():
    assert g.point_in_polygon(50, 50, SQUARE)
    assert not g.point_in_polygon(150, 50, SQUARE)
    assert not g.point_in_polygon(-1, -1, SQUARE)


def test_point_in_polygon_degenerate():
    assert not g.point_in_polygon(0, 0, [[0, 0], [1, 1]])
    assert not g.point_in_polygon(0, 0, [])


def test_point_in_rotated_polygon():
    diamond = [[50, 0], [100, 50], [50, 100], [0, 50]]
    assert g.point_in_polygon(50, 50, diamond)
    assert not g.point_in_polygon(5, 5, diamond)


def test_normalize_angle_deg_range():
    assert g.normalize_angle_deg(0) == 0
    assert g.normalize_angle_deg(90) == -90
    assert g.normalize_angle_deg(135) == -45
    assert g.normalize_angle_deg(-135) == 45
    for a in (-360, -179, -1, 0, 1, 89, 179, 360, 720):
        n = g.normalize_angle_deg(a)
        assert -90.0 <= n < 90.0


def test_polygon_long_edge_angle():
    # Wide rectangle: longest edge is horizontal -> ~0 degrees, length 200.
    rect = [[0, 0], [200, 0], [200, 50], [0, 50]]
    angle, length = g.polygon_long_edge_angle(rect)
    assert abs(length - 200.0) < 1e-6
    assert abs(angle) < 1e-6


def test_polygon_long_edge_angle_too_few_points():
    assert g.polygon_long_edge_angle([[0, 0], [1, 1]]) == (None, 0.0)
    assert g.polygon_long_edge_angle(None) == (None, 0.0)


def test_rect_iou():
    assert g.rect_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert g.rect_iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0
    # Half-overlap on x: intersection 50, union 150 -> 1/3.
    assert abs(g.rect_iou((0, 0, 10, 10), (5, 0, 10, 10)) - (50 / 150)) < 1e-6


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    raise SystemExit(1 if failures else 0)

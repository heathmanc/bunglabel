"""Unit tests for the pure review / quantity logic.

These run without PySide6 or OpenCV, so they are safe in headless CI.
Run with: python -m pytest tests/  (or python tests/test_review.py)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bung_labeler.core import review as r


def _battery(x: float) -> dict:
    return {
        "label": "battery",
        "class_id": 0,
        "kind": "obb",
        "points": [[x, 0], [x + 100, 0], [x + 100, 100], [x, 100]],
    }


def _bung(x: float, y: float) -> dict:
    return {"label": "bung", "class_id": 1, "x": x, "y": y, "w": 4, "h": 4}


def _full_battery(x: float, n: int = 6) -> list[dict]:
    return [_battery(x)] + [_bung(x + 10 + i * 10, 50) for i in range(n)]


def test_review_marker_accepts_only_this_tool():
    assert r.annotation_reviewed({"review": {"reviewed": True, "tool": "BungVision Label Studio"}})
    assert not r.annotation_reviewed({"reviewed": True, "review_status": "ok"})
    assert not r.annotation_reviewed({"reviewed": True})
    assert not r.annotation_reviewed(None)


def test_force_reviewed_detection():
    forced = {"review": {"reviewed": True, "tool": "BungVision Label Studio", "review_status": "forced_reviewed"}}
    assert r.annotation_force_reviewed(forced)
    legacy = {"review": {"reviewed": True, "reviewed_by": "BungVision Label Studio", "force_reviewed": True}}
    assert r.annotation_force_reviewed(legacy)
    plain = {"review": {"reviewed": True, "tool": "BungVision Label Studio"}}
    assert not r.annotation_force_reviewed(plain)


def test_review_record_is_version_stamped():
    rec = r.make_review_record("save_labels")
    assert rec["reviewed"] is True
    assert "BungVision Label Studio" in rec["reviewed_by"]
    assert "forced_review" not in rec

    frec = r.make_review_record("x", force=True, counts=(1, 5), expected=6)
    assert frec["review_status"] == "forced_reviewed"
    assert frec["battery_count"] == 1 and frec["bung_count"] == 5 and frec["expected_bungs"] == 6


def test_simple_label_collapses_variants():
    assert r.simple_label("battery_modelA", -1) == ("battery", 0)
    assert r.simple_label("bung_xl", -1) == ("bung", 1)
    assert r.simple_label("", 2) == ("retainer", 2)
    assert r.simple_label("positive_terminal", 7) == ("positive_terminal", 7)


def test_two_full_batteries_pass_without_force():
    boxes = _full_battery(0) + _full_battery(200)
    counts, outside = r.per_battery_bung_counts(boxes)
    assert counts == [6, 6]
    assert outside == 0
    assert r.quantities_satisfied(boxes, 6)


def test_missing_bung_requires_force():
    boxes = _full_battery(0) + _full_battery(200, n=5)
    assert not r.quantities_satisfied(boxes, 6)


def test_stray_bung_outside_requires_force():
    boxes = _full_battery(0) + [_bung(1000, 1000)]
    assert not r.quantities_satisfied(boxes, 6)


def test_no_battery_requires_force():
    boxes = [_bung(10, 10), _bung(20, 20)]
    assert not r.quantities_satisfied(boxes, 6)
    assert "at least 1" in r.quantity_summary_text(boxes, 6)


def test_counts_from_boxes():
    boxes = _full_battery(0) + _full_battery(200)
    assert r.counts_from_boxes(boxes) == (2, 12)


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

"""Unit tests for per-class label statistics."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bung_labeler.core import class_stats as cs


def test_count_labels_by_name():
    boxes = [{"label": "battery"}, {"label": "bung"}, {"label": "bung"}]
    assert cs.count_labels(boxes) == {"battery": 1, "bung": 2}


def test_box_label_fallbacks():
    assert cs.box_label({"label": "battery"}) == "battery"
    assert cs.box_label({"class_id": 3}) == "class_3"
    assert cs.box_label({}) == "unknown"


def test_merge_and_format():
    a = cs.count_labels([{"label": "bung"}])
    b = cs.count_labels([{"label": "bung"}, {"label": "battery"}])
    cs.merge_counts(a, b)
    assert a == {"bung": 2, "battery": 1}
    assert cs.format_counts(a) == "battery: 1, bung: 2"
    assert cs.format_counts({}) == "no labels"


if __name__ == "__main__":
    import traceback
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception:
                failures += 1; print(f"FAIL {name}"); traceback.print_exc()
    raise SystemExit(1 if failures else 0)

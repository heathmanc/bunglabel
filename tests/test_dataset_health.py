"""Unit tests for the dataset-health classification/tally helpers."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bung_labeler.core import dataset_health as dh


TOOL = "BungVision Label Studio"


def _battery(x: float) -> dict:
    return {"label": "battery", "class_id": 0, "kind": "obb",
            "points": [[x, 0], [x + 100, 0], [x + 100, 100], [x, 100]]}


def _bung(x: float) -> dict:
    return {"label": "bung", "class_id": 1, "x": x, "y": 50, "w": 4, "h": 4}


def _full(x: float, n: int = 6) -> list[dict]:
    return [_battery(x)] + [_bung(x + 10 + i * 10) for i in range(n)]


def _reviewed(boxes, forced=False):
    review = {"reviewed": True, "tool": TOOL}
    if forced:
        review["review_status"] = "forced_reviewed"
    return {"boxes": boxes, "review": review}


def test_status_unlabeled_and_empty():
    assert dh.annotation_status(None, 6) == "unlabeled"
    assert dh.annotation_status({"boxes": []}, 6) == "empty"


def test_status_needs_review():
    assert dh.annotation_status({"boxes": _full(0)}, 6) == "needs_review"


def test_status_ready_and_problem():
    assert dh.annotation_status(_reviewed(_full(0)), 6) == "ready"
    assert dh.annotation_status(_reviewed(_full(0, n=5)), 6) == "problem"


def test_status_forced():
    assert dh.annotation_status(_reviewed(_full(0, n=5), forced=True), 6) == "forced"


def test_unconstrained_ready_for_any_labels():
    # A lone widget would be a "problem" under the battery/bung check, but a
    # free-form (unconstrained) recipe accepts any reviewed labels as ready.
    data = _reviewed([{"label": "widget", "class_id": 5, "x": 0, "y": 0, "w": 4, "h": 4}])
    assert dh.annotation_status(data, 6, constrained=False) == "ready"


def test_tally_and_export_ready():
    statuses = ["ready", "ready", "forced", "problem", "needs_review", "empty", "unlabeled"]
    t = dh.tally_statuses(statuses)
    assert t["images"] == 7
    assert t["ready"] == 2 and t["forced"] == 1
    assert t["labeled"] == 5  # ready+ready+forced+problem+needs_review
    assert dh.export_ready(t) == 3


def test_merge_tally():
    a = dh.tally_statuses(["ready"])
    b = dh.tally_statuses(["problem", "ready"])
    dh.merge_tally(a, b)
    assert a["images"] == 3
    assert a["ready"] == 2
    assert a["problem"] == 1


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

"""Unit tests for the pure bulk-relabel logic (headless-safe)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bung_labeler.core import relabel as rl


def _sidecar(tmp, name, boxes, reviewed=False):
    data = {
        "image": f"/captures/{name}.jpg",
        "width": 1000,
        "height": 1000,
        "classes": ["battery", "bung", "retainer"],
        "boxes": boxes,
    }
    if reviewed:
        data["reviewed"] = True
        data["review_status"] = "reviewed"
        data["review"] = {"reviewed": True, "tool": "BungVision Label Studio"}
    (tmp / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


def test_remap_boxes_by_label():
    boxes = [
        {"label": "rubber_bung", "class_id": 5},
        {"label": "battery", "class_id": 0},
    ]
    out, changed = rl.remap_boxes(boxes, match_label="rubber_bung", new_label="bung", new_class_id=1)
    assert changed == 1
    assert out[0] == {"label": "bung", "class_id": 1}
    assert out[1] == {"label": "battery", "class_id": 0}


def test_remap_boxes_by_class_id():
    boxes = [{"label": "x", "class_id": 3}, {"label": "y", "class_id": 1}]
    out, changed = rl.remap_boxes(boxes, match_class_id=3, new_class_id=1, new_label="bung")
    assert changed == 1 and out[0]["class_id"] == 1 and out[0]["label"] == "bung"


def test_remap_requires_a_criterion():
    boxes = [{"label": "bung", "class_id": 1}]
    out, changed = rl.remap_boxes(boxes, new_label="x")
    assert changed == 0 and out[0]["label"] == "bung"


def test_scan_is_non_destructive(tmp_path):
    _sidecar(tmp_path, "a", [{"label": "bung", "class_id": 1}, {"label": "battery", "class_id": 0}])
    _sidecar(tmp_path, "b", [{"label": "battery", "class_id": 0}])
    report = rl.scan_relabel(tmp_path, match_label="bung", new_label="cap", new_class_id=4)
    assert report["images"] == 1 and report["boxes"] == 1
    # File unchanged on disk after a scan.
    data = json.loads((tmp_path / "a.json").read_text())
    assert data["boxes"][0]["label"] == "bung"


def test_apply_changes_and_clears_review(tmp_path):
    _sidecar(tmp_path, "a", [{"label": "bung", "class_id": 1}], reviewed=True)
    _sidecar(tmp_path, "b", [{"label": "battery", "class_id": 0}], reviewed=True)
    report = rl.apply_relabel(tmp_path, match_label="bung", new_label="cap", new_class_id=4)
    assert report["images"] == 1 and report["boxes"] == 1

    a = json.loads((tmp_path / "a.json").read_text())
    assert a["boxes"][0] == {"label": "cap", "class_id": 4}
    assert a["reviewed"] is False
    assert a["review_status"] == "needs_review"
    assert "review" not in a

    # Untouched image keeps its review marker.
    b = json.loads((tmp_path / "b.json").read_text())
    assert b["reviewed"] is True


def test_apply_can_preserve_review(tmp_path):
    _sidecar(tmp_path, "a", [{"label": "bung", "class_id": 1}], reviewed=True)
    rl.apply_relabel(tmp_path, match_label="bung", new_label="cap", new_class_id=4, clear_review=False)
    a = json.loads((tmp_path / "a.json").read_text())
    assert a["boxes"][0]["label"] == "cap"
    assert a["reviewed"] is True


if __name__ == "__main__":
    import tempfile
    import traceback
    from pathlib import Path

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                import inspect
                if "tmp_path" in inspect.signature(fn).parameters:
                    with tempfile.TemporaryDirectory() as d:
                        fn(Path(d))
                else:
                    fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    raise SystemExit(1 if failures else 0)

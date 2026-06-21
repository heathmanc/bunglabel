"""Unit tests for the pure export-report diagnostics (headless-safe)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bung_labeler.core import export_report as er

OBB_HEADER = "split,recipe,image,obb_count,battery_count,bung_count,retainer_count,class_mode"
DETECT_HEADER = "split,recipe,image,box_count,battery_count,bung_count,retainer_count,class_mode"

DATA_YAML = """path: /x
train: images/train
val: images/val
names:
  0: battery_a
  1: bung
  2: retainer_a
"""


def _write(tmp, manifest_lines=None, data_yaml=None):
    if manifest_lines is not None:
        (tmp / "manifest.csv").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    if data_yaml is not None:
        (tmp / "data.yaml").write_text(data_yaml, encoding="utf-8")
    return tmp


def test_class_names_parsed_in_order(tmp_path):
    _write(tmp_path, data_yaml=DATA_YAML)
    assert er.class_names(tmp_path) == ["battery_a", "bung", "retainer_a"]


def test_class_names_missing_yaml(tmp_path):
    assert er.class_names(tmp_path) == []


def test_missing_manifest(tmp_path):
    assert "No manifest.csv" in er.count_summary(tmp_path)


def test_empty_manifest(tmp_path):
    _write(tmp_path, manifest_lines=[OBB_HEADER])
    assert "No labeled images" in er.count_summary(tmp_path)


def test_obb_summary_totals(tmp_path):
    rows = [
        OBB_HEADER,
        "train,recipeA,recipeA__a.jpg,7,1,6,0,label_names",
        "val,recipeA,recipeA__b.jpg,6,1,5,0,label_names",
    ]
    _write(tmp_path, manifest_lines=rows, data_yaml=DATA_YAML)
    out = er.count_summary(tmp_path)
    assert "Images written: 2  (train 1, val 1)" in out
    assert "Labels written: 13" in out
    assert "Batteries: 2" in out
    assert "Bungs:     11" in out
    assert "Classes (3): battery_a, bung, retainer_a" in out
    # single recipe -> no per-recipe block
    assert "Images per recipe" not in out


def test_detect_label_column_and_empty_and_multi_recipe(tmp_path):
    rows = [
        DETECT_HEADER,
        "train,recipeA,a.jpg,2,1,1,0,model_specific",
        "train,recipeB,b.jpg,0,0,0,0,model_specific",
    ]
    _write(tmp_path, manifest_lines=rows)
    out = er.count_summary(tmp_path)
    assert "Labels written: 2" in out
    assert "Images with no usable labels: 1" in out
    assert "Images per recipe:" in out
    assert "recipeA: 1" in out and "recipeB: 1" in out


if __name__ == "__main__":
    import tempfile
    import traceback
    from pathlib import Path

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    raise SystemExit(1 if failures else 0)

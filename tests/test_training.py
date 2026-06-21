"""Unit tests for the pure training command builder/validator (headless-safe)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bung_labeler.core import training as t


def _params(tmp_path, **over):
    data = tmp_path / "data.yaml"
    data.write_text("names: [battery, bung]\n", encoding="utf-8")
    p = t.default_params()
    p["data"] = str(data)
    p.update(over)
    return p


def test_defaults_are_obb():
    assert t.default_params()["task"] == "obb"


def test_validate_clean(tmp_path):
    assert t.validate_train_params(_params(tmp_path)) == []


def test_validate_missing_data():
    errors = t.validate_train_params(t.default_params())
    assert any("Data YAML is required" in e for e in errors)


def test_validate_missing_data_file(tmp_path):
    p = _params(tmp_path, data=str(tmp_path / "nope.yaml"))
    assert any("not found" in e for e in t.validate_train_params(p))


def test_validate_bad_task(tmp_path):
    p = _params(tmp_path, task="banana")
    assert any("Task must be one of" in e for e in t.validate_train_params(p))


def test_validate_bad_batch(tmp_path):
    assert any("batch" in e for e in t.validate_train_params(_params(tmp_path, batch=0)))
    # -1 auto-batch is allowed.
    assert t.validate_train_params(_params(tmp_path, batch=-1)) == []


def test_validate_imgsz_range(tmp_path):
    assert any("imgsz" in e for e in t.validate_train_params(_params(tmp_path, imgsz=16)))


def test_build_command_basic(tmp_path):
    p = _params(tmp_path, imgsz=640, batch=8, epochs=50, device="0", name="run1")
    cmd = t.build_train_command("yolo", p)
    assert cmd[:3] == ["yolo", "obb", "train"]
    assert f"data={p['data']}" in cmd
    assert "imgsz=640" in cmd and "batch=8" in cmd and "epochs=50" in cmd
    assert "device=0" in cmd and "name=run1" in cmd
    assert "model=yolo11s-obb.pt" in cmd


def test_build_command_omits_empty_device_and_resume(tmp_path):
    p = _params(tmp_path, device="", resume=False)
    cmd = t.build_train_command("yolo", p)
    assert not any(c.startswith("device=") for c in cmd)
    assert not any(c.startswith("resume=") for c in cmd)


def test_build_command_custom_exe_and_resume(tmp_path):
    p = _params(tmp_path, resume=True)
    cmd = t.build_train_command("/opt/venv/bin/yolo", p)
    assert cmd[0] == "/opt/venv/bin/yolo"
    assert "resume=True" in cmd


_RESULTS_CSV = (
    "epoch,train/box_loss,train/cls_loss,metrics/mAP50(B),metrics/mAP50-95(B)\n"
    "1,1.5,2.0,0.10,0.05\n"
    "2,1.2,1.7,0.30,0.15\n"
    "3,1.0,1.5,0.55,0.32\n"
)


def test_parse_results_csv():
    rows = t.parse_results_csv(_RESULTS_CSV)
    assert len(rows) == 3
    assert rows[0]["epoch"] == 1.0
    assert rows[2]["train/box_loss"] == 1.0


def test_parse_results_csv_skips_malformed():
    bad = _RESULTS_CSV + "4,oops\n"
    rows = t.parse_results_csv(bad)
    assert len(rows) == 3


def test_parse_results_csv_empty():
    assert t.parse_results_csv("") == []
    assert t.parse_results_csv("epoch,train/box_loss\n") == []


def test_metric_series_by_substring():
    rows = t.parse_results_csv(_RESULTS_CSV)
    assert t.metric_series(rows, "box_loss") == [1.5, 1.2, 1.0]
    assert t.metric_series(rows, "mAP50-95") == [0.05, 0.15, 0.32]
    assert t.metric_series(rows, "nonexistent") == []


def test_chart_series_present_only():
    rows = t.parse_results_csv(_RESULTS_CSV)
    series = t.chart_series(rows)
    assert set(series) == {"box_loss", "cls_loss", "mAP50", "mAP50-95"}
    assert series["mAP50"] == [0.10, 0.30, 0.55]


_RESULTS_CSV_VAL = (
    "epoch,train/box_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
    "1,1.5,0.40,0.30,0.20,0.10\n"
    "2,1.2,0.70,0.60,0.65,0.40\n"
    "3,1.0,0.60,0.55,0.50,0.35\n"
)


def test_summarize_results_final_and_best():
    rows = t.parse_results_csv(_RESULTS_CSV_VAL)
    s = t.summarize_results(rows)
    assert s["epochs"] == 3 and s["rows"] == 3
    # Final row is epoch 3.
    assert s["final"]["mAP50-95"] == 0.35
    assert s["final"]["precision"] == 0.60
    # Best by mAP50-95 is epoch 2.
    assert s["best_epoch"] == 2
    assert s["best"]["mAP50-95"] == 0.40
    assert s["best"]["mAP50"] == 0.65


def test_summarize_results_empty():
    s = t.summarize_results([])
    assert s == {"epochs": 0, "rows": 0, "final": {}, "best": {}, "best_epoch": 0}


def test_format_duration():
    assert t.format_duration(47) == "47s"
    assert t.format_duration(312) == "5m 12s"
    assert t.format_duration(3784) == "1h 3m 4s"


if __name__ == "__main__":
    import tempfile
    import traceback
    import inspect
    from pathlib import Path

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
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

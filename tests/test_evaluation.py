"""Unit tests for the pure evaluation helpers (headless-safe)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bung_labeler.core import evaluation as ev


def _params(tmp_path, **over):
    model = tmp_path / "best.pt"; model.write_text("x", encoding="utf-8")
    data = tmp_path / "data.yaml"; data.write_text("names: [battery, bung]\n", encoding="utf-8")
    p = ev.default_eval_params()
    p.update({"model": str(model), "data": str(data)})
    p.update(over)
    return p


def test_validate_clean(tmp_path):
    assert ev.validate_eval_params(_params(tmp_path)) == []


def test_validate_missing_model(tmp_path):
    p = _params(tmp_path, model=str(tmp_path / "nope.pt"))
    assert any("Model not found" in e for e in ev.validate_eval_params(p))


def test_validate_bad_split(tmp_path):
    assert any("Split must be one of" in e for e in ev.validate_eval_params(_params(tmp_path, split="holdout")))


def test_validate_bad_imgsz(tmp_path):
    assert any("imgsz" in e for e in ev.validate_eval_params(_params(tmp_path, imgsz=0)))


def test_build_command(tmp_path):
    p = _params(tmp_path, imgsz=640, device="0", split="test")
    cmd = ev.build_eval_command("/usr/bin/python", p)
    assert cmd[:3] == ["/usr/bin/python", "-m", ev.RUNNER_MODULE]
    assert "--model" in cmd and p["model"] in cmd
    assert "--imgsz" in cmd and "640" in cmd
    assert "--split" in cmd and "test" in cmd
    assert "--device" in cmd and "0" in cmd


def test_build_command_omits_empty_device(tmp_path):
    cmd = ev.build_eval_command("python", _params(tmp_path, device=""))
    assert "--device" not in cmd


def test_parse_metrics_with_console_noise():
    metrics = {"map50": 0.95, "map": 0.71, "mp": 0.93, "mr": 0.9,
               "classes": [{"name": "battery", "precision": 0.98, "recall": 0.97, "map50": 0.99}]}
    text = (
        "Ultralytics 8.x ... scanning images\n"
        + ev.METRICS_START + "\n" + json.dumps(metrics) + "\n" + ev.METRICS_END + "\n"
        + "Results saved to runs/val\n"
    )
    parsed = ev.parse_metrics_output(text)
    assert parsed["map50"] == 0.95
    assert parsed["classes"][0]["name"] == "battery"


def test_parse_metrics_absent_returns_none():
    assert ev.parse_metrics_output("no metrics here") is None
    assert ev.parse_metrics_output(ev.METRICS_START + " not json " + ev.METRICS_END) is None


def test_format_metrics_overall_and_classes():
    out = ev.format_metrics({"map50": 0.95, "map": 0.71, "mp": 0.93, "mr": 0.9,
                             "classes": [{"name": "bung", "precision": 0.91, "recall": 0.89, "map50": 0.93}]})
    assert "mAP50:    0.950" in out
    assert "bung: P 0.910" in out


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

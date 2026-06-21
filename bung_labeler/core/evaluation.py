"""Pure helpers for evaluating a trained YOLO model against a labeled split.

Builds/validates the evaluation command and parses the structured metrics the
runner emits, but does not run anything (the UI runs it via QProcess). Testable
without Qt/OpenCV/Ultralytics.

Evaluation runs ``bung_labeler.eval_runner`` (a thin Ultralytics wrapper) which
prints a JSON metrics block between sentinels so the result can be parsed
reliably instead of scraping the console table.
"""
from __future__ import annotations

import json
from pathlib import Path

VALID_TASKS = ("obb", "detect", "segment", "pose", "classify")
VALID_SPLITS = ("val", "test", "train")

METRICS_START = "<<<BUNGVISION_METRICS_JSON>>>"
METRICS_END = "<<<END_BUNGVISION_METRICS_JSON>>>"

RUNNER_MODULE = "bung_labeler.eval_runner"


def default_eval_params() -> dict:
    return {
        "task": "obb",
        "model": "",
        "data": "",
        "imgsz": 736,
        "device": "0",
        "split": "val",
    }


def validate_eval_params(params: dict) -> list[str]:
    """Return a list of human-readable problems. Empty == ready to run."""
    errors: list[str] = []

    task = str(params.get("task", "")).strip().lower()
    if task not in VALID_TASKS:
        errors.append(f"Task must be one of: {', '.join(VALID_TASKS)}.")

    model = str(params.get("model", "")).strip()
    if not model:
        errors.append("Model is required (the trained .pt checkpoint to evaluate).")
    elif not Path(model).exists():
        errors.append(f"Model not found: {model}")

    data = str(params.get("data", "")).strip()
    if not data:
        errors.append("Data YAML is required (the dataset to evaluate against).")
    elif not Path(data).exists():
        errors.append(f"Data YAML not found: {data}")

    split = str(params.get("split", "")).strip().lower()
    if split not in VALID_SPLITS:
        errors.append(f"Split must be one of: {', '.join(VALID_SPLITS)}.")

    try:
        v = int(params.get("imgsz"))
        if not (32 <= v <= 8192):
            errors.append("imgsz must be between 32 and 8192.")
    except (TypeError, ValueError):
        errors.append("imgsz must be an integer.")

    return errors


def build_eval_command(python_exe: str, params: dict, runner_module: str = RUNNER_MODULE) -> list[str]:
    """Build the argv to run the metrics runner as `python -m <runner_module>`."""
    cmd = [python_exe, "-m", runner_module,
           "--task", str(params.get("task", "obb")).strip().lower(),
           "--model", str(params.get("model", "")).strip(),
           "--data", str(params.get("data", "")).strip(),
           "--imgsz", str(int(params.get("imgsz", 736))),
           "--split", str(params.get("split", "val")).strip().lower()]
    device = str(params.get("device", "")).strip()
    if device:
        cmd += ["--device", device]
    return cmd


def parse_metrics_output(text: str) -> dict | None:
    """Extract the JSON metrics block the runner prints between sentinels."""
    if METRICS_START not in text or METRICS_END not in text:
        return None
    try:
        block = text.split(METRICS_START, 1)[1].split(METRICS_END, 1)[0].strip()
        data = json.loads(block)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def format_metrics(metrics: dict) -> str:
    """Render parsed metrics as an operator-readable summary."""
    if not metrics:
        return "No metrics were produced."
    lines = [
        "Overall:",
        f"  mAP50:    {float(metrics.get('map50', 0.0)):.3f}",
        f"  mAP50-95: {float(metrics.get('map', 0.0)):.3f}",
        f"  Precision:{float(metrics.get('mp', 0.0)):.3f}",
        f"  Recall:   {float(metrics.get('mr', 0.0)):.3f}",
    ]
    classes = metrics.get("classes") or []
    if classes:
        lines.append("")
        lines.append("Per class (precision / recall / mAP50):")
        for c in classes:
            name = str(c.get("name", "?"))
            lines.append(
                f"  {name}: P {float(c.get('precision', 0.0)):.3f}  "
                f"R {float(c.get('recall', 0.0)):.3f}  "
                f"mAP50 {float(c.get('map50', 0.0)):.3f}"
            )
    return "\n".join(lines)

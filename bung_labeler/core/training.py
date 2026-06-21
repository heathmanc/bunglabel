"""Pure helpers for launching Ultralytics YOLO training as a subprocess.

This module builds and validates the training command but does not run it (the
UI runs it via QProcess so it stays cancelable and non-blocking). Keeping the
command construction here makes it unit testable without Qt, OpenCV, or
Ultralytics installed.

The generated command targets the Ultralytics ``yolo`` CLI, e.g.:
    yolo obb train model=yolo11s-obb.pt data=data.yaml imgsz=736 batch=16 ...
"""
from __future__ import annotations

from pathlib import Path

VALID_TASKS = ("obb", "detect", "segment", "pose", "classify")


def default_params() -> dict:
    return {
        "task": "obb",
        "model": "yolo11s-obb.pt",
        "data": "",
        "imgsz": 736,
        "batch": 16,
        "epochs": 100,
        "device": "0",
        "project": "data/training",
        "name": "bungvision",
        "patience": 50,
        "workers": 8,
        "resume": False,
    }


def validate_train_params(params: dict) -> list[str]:
    """Return a list of human-readable problems. Empty == ready to run."""
    errors: list[str] = []

    task = str(params.get("task", "")).strip().lower()
    if task not in VALID_TASKS:
        errors.append(f"Task must be one of: {', '.join(VALID_TASKS)}.")

    model = str(params.get("model", "")).strip()
    if not model:
        errors.append("Base model is required (e.g. yolo11s-obb.pt or a .pt checkpoint).")

    data = str(params.get("data", "")).strip()
    if not data:
        errors.append("Data YAML is required. Export a dataset, then point to its data.yaml.")
    elif not Path(data).exists():
        errors.append(f"Data YAML not found: {data}")

    for key, lo, hi in (("imgsz", 32, 8192), ("epochs", 1, 100000), ("patience", 0, 100000), ("workers", 0, 256)):
        try:
            v = int(params.get(key))
        except (TypeError, ValueError):
            errors.append(f"{key} must be an integer.")
            continue
        if not (lo <= v <= hi):
            errors.append(f"{key} must be between {lo} and {hi}.")

    # batch may be -1 (Ultralytics auto-batch) or a positive integer.
    try:
        batch = int(params.get("batch"))
        if batch == 0 or batch < -1:
            errors.append("batch must be a positive integer, or -1 for auto.")
    except (TypeError, ValueError):
        errors.append("batch must be an integer (or -1 for auto).")

    if not str(params.get("name", "")).strip():
        errors.append("Run name is required.")

    return errors


def build_train_command(yolo_exe: str, params: dict) -> list[str]:
    """Build the argv list for the Ultralytics CLI from validated params.

    yolo_exe is the executable/entrypoint to invoke (default "yolo"); it is left
    overridable so a full path or wrapper can be supplied in environments where
    ``yolo`` is not on PATH.
    """
    task = str(params.get("task", "obb")).strip().lower()
    cmd = [yolo_exe or "yolo", task, "train"]

    def add(key: str, value) -> None:
        cmd.append(f"{key}={value}")

    add("model", str(params.get("model", "")).strip())
    add("data", str(params.get("data", "")).strip())
    add("imgsz", int(params.get("imgsz", 736)))
    add("batch", int(params.get("batch", 16)))
    add("epochs", int(params.get("epochs", 100)))
    add("patience", int(params.get("patience", 50)))
    add("workers", int(params.get("workers", 8)))

    device = str(params.get("device", "")).strip()
    if device:
        add("device", device)

    project = str(params.get("project", "")).strip()
    if project:
        add("project", project)
    name = str(params.get("name", "")).strip()
    if name:
        add("name", name)

    if params.get("resume"):
        add("resume", "True")

    return cmd


# --- Live training-metrics parsing -------------------------------------------
# Ultralytics writes <project>/<name>/results.csv, one row per finished epoch.
# Parsing that file (rather than the tqdm stdout, which uses carriage returns)
# gives a clean, pollable source for the live loss/mAP chart.

def parse_results_csv(text: str) -> list[dict]:
    """Parse an Ultralytics results.csv into a list of per-epoch row dicts.

    Numeric cells are converted to floats; non-numeric cells stay as strings.
    Malformed rows (wrong column count) are skipped so a half-written file
    being polled mid-train does not raise.
    """
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].split(",")]
    rows: list[dict] = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(header):
            continue
        row: dict = {}
        for key, raw in zip(header, parts):
            try:
                row[key] = float(raw)
            except ValueError:
                row[key] = raw
        rows.append(row)
    return rows


def _first_matching_column(columns: list[str], needle: str) -> str | None:
    needle = needle.lower()
    for col in columns:
        if needle in col.lower():
            return col
    return None


def metric_series(rows: list[dict], needle: str) -> list[float]:
    """Numeric series for the first column whose name contains ``needle``."""
    if not rows:
        return []
    col = _first_matching_column(list(rows[0].keys()), needle)
    if col is None:
        return []
    out: list[float] = []
    for r in rows:
        v = r.get(col)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


# Series shown on the live chart: a short label and the substring used to find
# the matching results.csv column across detect/obb/segment/pose runs.
CHART_SERIES = (
    ("box_loss", "train/box_loss"),
    ("cls_loss", "train/cls_loss"),
    ("mAP50", "metrics/mAP50("),
    ("mAP50-95", "metrics/mAP50-95("),
)


def chart_series(rows: list[dict]) -> dict[str, list[float]]:
    """Return {label: series} for the standard chart metrics that are present."""
    out: dict[str, list[float]] = {}
    for label, needle in CHART_SERIES:
        series = metric_series(rows, needle)
        if series:
            out[label] = series
    return out


# Validation metrics shown in the training-finished summary.
SUMMARY_METRICS = (
    ("precision", "metrics/precision"),
    ("recall", "metrics/recall"),
    ("mAP50", "metrics/mAP50("),
    ("mAP50-95", "metrics/mAP50-95("),
)


def summarize_results(rows: list[dict]) -> dict:
    """Summarize a parsed results.csv into final + best validation metrics.

    Returns a dict with:
      epochs       - epoch number of the last row (or row count if no column)
      rows         - number of recorded epochs
      final        - {metric: value} from the last epoch
      best         - {metric: value} from the best epoch (ranked by mAP50-95,
                     falling back to mAP50)
      best_epoch   - epoch number of that best row
    Empty rows yield {"epochs": 0, "rows": 0, "final": {}, "best": {}, "best_epoch": 0}.
    """
    if not rows:
        return {"epochs": 0, "rows": 0, "final": {}, "best": {}, "best_epoch": 0}

    epoch_series = metric_series(rows, "epoch")
    epochs = int(epoch_series[-1]) if epoch_series else len(rows)

    final: dict[str, float] = {}
    for label, needle in SUMMARY_METRICS:
        series = metric_series(rows, needle)
        if series:
            final[label] = series[-1]

    # Rank epochs by mAP50-95, then mAP50, to find the best checkpoint.
    rank = metric_series(rows, "mAP50-95(") or metric_series(rows, "mAP50(")
    best_idx = max(range(len(rank)), key=lambda i: rank[i]) if rank else len(rows) - 1

    best: dict[str, float] = {}
    for label, needle in SUMMARY_METRICS:
        series = metric_series(rows, needle)
        if series and best_idx < len(series):
            best[label] = series[best_idx]

    if epoch_series and best_idx < len(epoch_series):
        best_epoch = int(epoch_series[best_idx])
    else:
        best_epoch = best_idx + 1

    return {"epochs": epochs, "rows": len(rows), "final": final, "best": best, "best_epoch": best_epoch}


def format_duration(seconds: float) -> str:
    """Human duration like '1h 23m 4s' / '5m 12s' / '47s'."""
    seconds = int(max(0, round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

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

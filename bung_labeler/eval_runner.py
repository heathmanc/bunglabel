#!/usr/bin/env python3
"""Evaluate a trained YOLO model and emit structured metrics.

Run as a subprocess by the app's Evaluate action:
    python -m bung_labeler.eval_runner --task obb --model best.pt \
        --data data.yaml --imgsz 736 --split val --device 0

Ultralytics prints its own table to the console; this wrapper additionally
prints a JSON metrics block between sentinels so the app can parse it reliably.
Kept thin and import-light so it only needs Ultralytics at runtime.
"""
from __future__ import annotations

import argparse
import json
import sys

from bung_labeler.core.evaluation import METRICS_END, METRICS_START


def _build_metrics(metrics) -> dict:
    box = getattr(metrics, "box", None)
    names = getattr(metrics, "names", None) or {}
    out: dict = {
        "map50": float(getattr(box, "map50", 0.0) or 0.0),
        "map": float(getattr(box, "map", 0.0) or 0.0),
        "mp": float(getattr(box, "mp", 0.0) or 0.0),
        "mr": float(getattr(box, "mr", 0.0) or 0.0),
        "classes": [],
    }
    try:
        for i, c in enumerate(getattr(box, "ap_class_index", []) or []):
            p, r, ap50, ap = box.class_result(i)
            cname = names.get(int(c), str(c)) if isinstance(names, dict) else str(c)
            out["classes"].append({
                "name": cname,
                "precision": float(p),
                "recall": float(r),
                "map50": float(ap50),
                "map": float(ap),
            })
    except Exception:
        # Per-class breakdown is best-effort; overall metrics still report.
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="obb")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=736)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - depends on runtime env
        print(f"Could not import Ultralytics: {exc}", file=sys.stderr)
        return 2

    kwargs = {"data": args.data, "imgsz": args.imgsz, "split": args.split, "verbose": True}
    if args.device:
        kwargs["device"] = args.device

    model = YOLO(args.model)
    metrics = model.val(**kwargs)

    payload = _build_metrics(metrics)
    print(METRICS_START)
    print(json.dumps(payload))
    print(METRICS_END)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

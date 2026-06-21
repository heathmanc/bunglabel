"""Operator-readable diagnostics for an exported YOLO dataset.

Pure stdlib (csv + pathlib): reads back the files an export wrote (manifest.csv,
data.yaml) and summarizes what actually landed in the dataset. No Qt/OpenCV, so
it is unit testable headlessly.
"""
from __future__ import annotations

import csv
from pathlib import Path


def class_names(out: Path) -> list[str]:
    """Read the ordered class names from a dataset's data.yaml ``names:`` block."""
    data_yaml = Path(out) / "data.yaml"
    if not data_yaml.exists():
        return []
    names: list[str] = []
    in_names = False
    try:
        for raw in data_yaml.read_text(encoding="utf-8").splitlines():
            if raw.strip() == "names:":
                in_names = True
                continue
            if in_names:
                stripped = raw.strip()
                if not raw.startswith(" ") or not stripped:
                    break
                # Lines look like "  0: battery_model".
                _, _, name = stripped.partition(":")
                if name.strip():
                    names.append(name.strip())
    except Exception:
        return names
    return names


def _int(row: dict, key: str) -> int:
    try:
        return int(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def count_summary(out: Path) -> str:
    """Build an operator-readable breakdown of what an export wrote, from its
    manifest.csv: image counts per split/recipe, object totals
    (battery/bung/retainer), and the exported class list, so the operator can
    confirm the export matches expectations before training."""
    out = Path(out)
    manifest = out / "manifest.csv"
    if not manifest.exists():
        return "No manifest.csv was written; cannot summarize export counts."
    try:
        rows = list(csv.DictReader(manifest.read_text(encoding="utf-8").splitlines()))
    except Exception as exc:
        return f"Could not read manifest.csv: {exc}"
    if not rows:
        return "No labeled images were written to this dataset."

    # The detect exporter names its per-image label column box_count, the OBB
    # exporter uses obb_count. Accept either.
    label_key = "obb_count" if "obb_count" in rows[0] else "box_count"

    split_images = {"train": 0, "val": 0}
    per_recipe: dict[str, int] = {}
    totals = {"labels": 0, "battery": 0, "bung": 0, "retainer": 0}
    empty_images = 0
    for row in rows:
        split = str(row.get("split", "")).strip()
        if split in split_images:
            split_images[split] += 1
        recipe = str(row.get("recipe", "")).strip() or "(unknown)"
        per_recipe[recipe] = per_recipe.get(recipe, 0) + 1
        n_labels = _int(row, label_key)
        totals["labels"] += n_labels
        totals["battery"] += _int(row, "battery_count")
        totals["bung"] += _int(row, "bung_count")
        totals["retainer"] += _int(row, "retainer_count")
        if n_labels == 0:
            empty_images += 1

    total_images = len(rows)
    lines = [
        f"Images written: {total_images}  (train {split_images['train']}, val {split_images['val']})",
        f"Labels written: {totals['labels']}",
        f"  Batteries: {totals['battery']}",
        f"  Bungs:     {totals['bung']}",
        f"  Retainers: {totals['retainer']}",
    ]
    if empty_images:
        lines.append(f"Images with no usable labels: {empty_images}")
    if len(per_recipe) > 1:
        lines.append("Images per recipe:")
        for recipe in sorted(per_recipe):
            lines.append(f"  {recipe}: {per_recipe[recipe]}")

    classes = class_names(out)
    if classes:
        lines.append(f"Classes ({len(classes)}): " + ", ".join(classes))
    return "\n".join(lines)

"""Pure helpers for per-class label statistics.

Counts how many annotation boxes of each class exist so an operator can spot an
under-represented class before training.  No Qt/OpenCV here so it is unit
testable; the UI feeds in plain box dicts.
"""
from __future__ import annotations


def box_label(box: dict) -> str:
    """Human label for a box dict, falling back to class_<id> then 'unknown'."""
    name = str(box.get("label") or "").strip()
    if name:
        return name
    cid = box.get("class_id")
    if cid is not None:
        return f"class_{cid}"
    return "unknown"


def count_labels(boxes: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for b in boxes or []:
        name = box_label(b)
        counts[name] = counts.get(name, 0) + 1
    return counts


def merge_counts(into: dict[str, int], other: dict[str, int]) -> None:
    for name, n in other.items():
        into[name] = into.get(name, 0) + n


def format_counts(counts: dict[str, int]) -> str:
    """One-line, stable, human summary like 'battery: 2, bung: 12'."""
    if not counts:
        return "no labels"
    parts = [f"{name}: {counts[name]}" for name in sorted(counts)]
    return ", ".join(parts)

"""Pure review / quantity logic for BungVision Label Studio.

This module holds the operator-review decision logic and the bung-quantity
checks as plain functions with no Qt or OpenCV dependency, so they can be unit
tested without a display. Both the UI (``main_window``) and the exporter
(``yolo_export``) import from here, which keeps the review-marker rules in one
place instead of being duplicated and able to drift.
"""
from __future__ import annotations

import time

from ..version import APP_TITLE
from .geometry import point_in_polygon

_REVIEW_MARKER_KEYS = ("source", "tool", "review_source", "reviewed_by", "reviewer", "app")


# --- review markers -------------------------------------------------------

def is_label_studio_review_marker(review: dict | None) -> bool:
    """True when ``review`` is an operator-review marker created by this tool."""
    if not isinstance(review, dict) or not bool(review.get("reviewed", False)):
        return False
    text = " ".join(str(review.get(k, "")) for k in _REVIEW_MARKER_KEYS).lower()
    return (
        "bungvision_label_studio" in text
        or "bung label studio" in text
        or "label studio" in text
    )


def annotation_reviewed(data: dict | None) -> bool:
    """True only for labels explicitly reviewed inside this labeler.

    BungVision runtime/import JSON can contain generic fields such as
    ``reviewed=true`` or ``review_status=ok/pass``. Those must not count as
    operator review for training export. Legacy v0.9.28-v0.9.30 Label Studio
    markers are still accepted because they include ``reviewed_by`` containing
    "BungVision Label Studio".
    """
    if not data:
        return False
    review = data.get("review") if isinstance(data, dict) else None
    if is_label_studio_review_marker(review):
        return True
    if bool(data.get("reviewed", False)):
        top_level_review = {
            "reviewed": True,
            "source": data.get("review_source") or data.get("source") or data.get("origin") or data.get("imported_from"),
            "tool": data.get("review_tool") or data.get("tool") or data.get("app"),
            "reviewed_by": data.get("reviewed_by"),
        }
        return is_label_studio_review_marker(top_level_review)
    return False


def annotation_force_reviewed(data: dict | None) -> bool:
    """True when an image was force-reviewed despite a quantity mismatch."""
    if not data or not annotation_reviewed(data):
        return False
    review = data.get("review") if isinstance(data, dict) else None
    if isinstance(review, dict) and (
        bool(review.get("forced_review", False))
        or bool(review.get("force_reviewed", False))
        or str(review.get("review_status", "")).lower() == "forced_reviewed"
    ):
        return True
    return bool(data.get("forced_review", False) or data.get("force_reviewed", False))


def make_review_record(
    reason: str = "operator_review",
    *,
    force: bool = False,
    counts: tuple[int, int] = (0, 0),
    expected: int = 0,
) -> dict:
    """Build the review sidecar stamp written when an image is reviewed."""
    record = {
        "reviewed": True,
        "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reviewed_by": APP_TITLE,
        "source": "bungvision_label_studio",
        "tool": APP_TITLE.split(" v")[0],
        "reason": reason,
    }
    if force:
        batt, bung = counts
        record.update({
            "forced_review": True,
            "review_status": "forced_reviewed",
            "forced_reason": "quantity_mismatch",
            "battery_count": int(batt),
            "bung_count": int(bung),
            "expected_bungs": int(expected),
            "warning": "Operator force-reviewed this image even though the required quantities did not match.",
        })
    return record


# --- box classification ---------------------------------------------------

def simple_label(label: str, class_id: int = -1) -> tuple[str, int]:
    """Collapse any battery/bung/retainer variant to its canonical (label, id)."""
    label_l = str(label or "").lower()
    if label_l == "battery" or label_l.startswith("battery_") or int(class_id) == 0:
        return "battery", 0
    if label_l == "bung" or label_l.startswith("bung_") or int(class_id) == 1:
        return "bung", 1
    if label_l == "retainer" or label_l.startswith("retainer_") or int(class_id) == 2:
        return "retainer", 2
    return str(label or ""), int(class_id)


def normalize_box(box: dict) -> dict:
    """Normalize a BungVision runtime JSON box to the editor's simple labels."""
    original_label = str(box.get("label", "") or "")
    original_class_id = int(box.get("class_id", -1))
    label, class_id = simple_label(original_label, original_class_id)

    normalized = dict(box)
    normalized["label"] = label
    normalized["class_id"] = class_id

    if "source_label" not in normalized and original_label != label:
        normalized["source_label"] = original_label
    if "source_class_id" not in normalized and original_class_id != class_id:
        normalized["source_class_id"] = original_class_id

    return normalized


def _box_is_battery(box: dict) -> bool:
    return str(box.get("label", "")).startswith("battery") or int(box.get("class_id", -1)) == 0


def _box_is_bung(box: dict) -> bool:
    return str(box.get("label", "")).startswith("bung") or int(box.get("class_id", -1)) == 1


# --- geometry (dependency-free) -------------------------------------------

def box_polygon(box: dict) -> list[list[float]]:
    """Return four image-space corner points for either an OBB or a plain box."""
    pts = box.get("points") or box.get("obb") or []
    if len(pts) >= 4:
        return [[float(p[0]), float(p[1])] for p in pts[:4]]
    x = float(box.get("x", 0.0))
    y = float(box.get("y", 0.0))
    w = float(box.get("w", 0.0))
    h = float(box.get("h", 0.0))
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def box_center(box: dict) -> tuple[float, float]:
    poly = box_polygon(box)
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    return cx, cy


# --- quantity checks ------------------------------------------------------

def counts_from_boxes(boxes: list[dict]) -> tuple[int, int]:
    """Return (battery_count, bung_count) for a list of raw box dicts."""
    batt = bung = 0
    for raw in boxes:
        b = normalize_box(raw)
        if _box_is_battery(b):
            batt += 1
        elif _box_is_bung(b):
            bung += 1
    return batt, bung


def per_battery_bung_counts(boxes: list[dict]) -> tuple[list[int], int]:
    """Assign each bung to the battery whose polygon contains its center.

    Returns (per-battery bung counts, number of bungs outside every battery).
    Supports any number of batteries in view, not just one.
    """
    batteries: list[list[list[float]]] = []
    bung_centers: list[tuple[float, float]] = []
    for raw in boxes:
        b = normalize_box(raw)
        if _box_is_battery(b):
            batteries.append(box_polygon(b))
        elif _box_is_bung(b):
            bung_centers.append(box_center(b))

    counts = [0] * len(batteries)
    outside = 0
    for cx, cy in bung_centers:
        assigned = False
        for i, poly in enumerate(batteries):
            if point_in_polygon(cx, cy, poly):
                counts[i] += 1
                assigned = True
                break
        if not assigned:
            outside += 1
    return counts, outside


def quantities_satisfied(boxes: list[dict], expected: int) -> bool:
    """Review passes without force when every battery in view holds exactly the
    expected number of bungs, there is at least one battery, and no bung falls
    outside all batteries. This lets multiple fully-labeled batteries pass."""
    counts, outside = per_battery_bung_counts(boxes)
    if not counts or outside:
        return False
    return all(c == int(expected) for c in counts)


def quantity_summary_text(boxes: list[dict], expected: int) -> str:
    """Human-readable per-battery breakdown for review dialogs."""
    counts, outside = per_battery_bung_counts(boxes)
    if not counts:
        return f"Batteries: 0 (need at least 1)\nExpected bungs per battery: {expected}"
    lines = [f"Battery {i + 1}: {c} / {expected} bungs" for i, c in enumerate(counts)]
    if outside:
        lines.append(f"Bungs outside any battery: {outside}")
    return "\n".join(lines)

"""Pure helpers for the dataset health dashboard.

Given saved annotation dicts these functions classify each image and tally the
totals shown in the dashboard. They contain no Qt/OpenCV/file-IO so they stay
unit testable; the UI walks the recipe folders and feeds the dicts in.
"""
from __future__ import annotations

from . import review as review_logic

# Per-image classifications.  "labeled" statuses are the ones that carry boxes.
LABELED_STATUSES = ("ready", "forced", "problem", "needs_review")
ALL_STATUSES = LABELED_STATUSES + ("empty", "unlabeled")


def annotation_status(data: dict | None, expected: int, constrained: bool = True) -> str:
    """Classify one image from its sidecar annotation dict.

    Returns one of:
      unlabeled    - no sidecar JSON at all
      empty        - sidecar exists but has no boxes
      needs_review - has boxes but not marked reviewed
      forced       - reviewed via Force Review (intentional mismatch)
      ready        - reviewed and the quantity check passes (export ready)
      problem      - reviewed but the quantity check fails
    """
    if data is None:
        return "unlabeled"
    boxes = data.get("boxes") or []
    if not boxes:
        return "empty"
    if not review_logic.annotation_reviewed(data):
        return "needs_review"
    if review_logic.annotation_force_reviewed(data):
        return "forced"
    exp = int(expected) if constrained else 0
    if review_logic.quantities_satisfied(boxes, exp):
        return "ready"
    return "problem"


def new_tally() -> dict:
    t = {"images": 0, "labeled": 0}
    for s in ALL_STATUSES:
        t[s] = 0
    return t


def add_status(tally: dict, status: str) -> None:
    """Fold one classification into a running tally (does not count images)."""
    tally[status] = tally.get(status, 0) + 1
    if status in LABELED_STATUSES:
        tally["labeled"] += 1


def tally_statuses(statuses: list[str]) -> dict:
    t = new_tally()
    for s in statuses:
        t["images"] += 1
        add_status(t, s)
    return t


def export_ready(tally: dict) -> int:
    """Images that would be included in a reviewed-only export."""
    return int(tally.get("ready", 0)) + int(tally.get("forced", 0))


def merge_tally(into: dict, other: dict) -> None:
    for k, v in other.items():
        into[k] = into.get(k, 0) + v

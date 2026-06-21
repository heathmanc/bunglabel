"""Bulk class relabeling across a recipe's saved annotation sidecars.

Pure stdlib (json + pathlib): walks the per-recipe label folder, remaps boxes
that match a source class to a target class, and (by default) clears the review
marker on any changed image so it must be re-reviewed before it can re-enter
reviewed-only export. No Qt/OpenCV, so it is unit testable headlessly.

Clearing review on change is the training-safe choice: relabeling can change the
battery/bung counts an image was reviewed against, so a previously "reviewed OK"
image should not silently stay eligible for training with different classes.
"""
from __future__ import annotations

import json
from pathlib import Path

# Mirrors the review keys that storage.save_annotations manages, so a changed
# image is returned to the unreviewed queue.
_REVIEW_KEYS = (
    "review",
    "reviewed",
    "reviewed_at",
    "reviewed_by",
    "review_status",
    "review_source",
    "review_tool",
    "forced_review",
    "force_reviewed",
)


def _box_matches(box: dict, match_label: str | None, match_class_id: int | None) -> bool:
    """A box matches when it satisfies every provided criterion (AND)."""
    if match_label is None and match_class_id is None:
        return False
    if match_label is not None:
        if str(box.get("label", "")).strip().lower() != match_label.strip().lower():
            return False
    if match_class_id is not None:
        if int(box.get("class_id", -1)) != int(match_class_id):
            return False
    return True


def remap_boxes(
    boxes: list[dict],
    *,
    match_label: str | None = None,
    match_class_id: int | None = None,
    new_label: str | None = None,
    new_class_id: int | None = None,
) -> tuple[list[dict], int]:
    """Return (updated_boxes, changed_count). Non-matching boxes pass through."""
    out: list[dict] = []
    changed = 0
    for b in boxes:
        nb = dict(b)
        if _box_matches(nb, match_label, match_class_id):
            if new_label is not None:
                nb["label"] = new_label
            if new_class_id is not None:
                nb["class_id"] = int(new_class_id)
            changed += 1
        out.append(nb)
    return out, changed


def _iter_sidecars(label_dir: Path):
    for p in sorted(Path(label_dir).glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            yield p, data


def scan_relabel(
    label_dir: Path,
    *,
    match_label: str | None = None,
    match_class_id: int | None = None,
    new_label: str | None = None,
    new_class_id: int | None = None,
) -> dict:
    """Dry run: report how many boxes/images a relabel would change."""
    images = 0
    total = 0
    files: list[str] = []
    for p, data in _iter_sidecars(label_dir):
        _boxes, changed = remap_boxes(
            data.get("boxes", []),
            match_label=match_label,
            match_class_id=match_class_id,
            new_label=new_label,
            new_class_id=new_class_id,
        )
        if changed:
            images += 1
            total += changed
            files.append(p.name)
    return {"images": images, "boxes": total, "files": files}


def apply_relabel(
    label_dir: Path,
    *,
    match_label: str | None = None,
    match_class_id: int | None = None,
    new_label: str | None = None,
    new_class_id: int | None = None,
    clear_review: bool = True,
) -> dict:
    """Apply the relabel in place, writing only the sidecars that changed.

    Returns the same report shape as scan_relabel. By default the review marker
    on changed images is cleared so they re-enter the review queue.
    """
    images = 0
    total = 0
    files: list[str] = []
    for p, data in _iter_sidecars(label_dir):
        new_boxes, changed = remap_boxes(
            data.get("boxes", []),
            match_label=match_label,
            match_class_id=match_class_id,
            new_label=new_label,
            new_class_id=new_class_id,
        )
        if not changed:
            continue
        data["boxes"] = new_boxes
        if clear_review:
            for key in _REVIEW_KEYS:
                data.pop(key, None)
            data["reviewed"] = False
            data["review_status"] = "needs_review"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        images += 1
        total += changed
        files.append(p.name)
    return {"images": images, "boxes": total, "files": files}

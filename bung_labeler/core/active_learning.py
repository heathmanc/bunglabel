"""Active-learning prioritization for BungVision Label Studio.

Pure scoring logic (no Qt/OpenCV) for ranking unreviewed images by how much a
model's detections disagree with the recipe expectation. Images the model is
unsure about, or where the detected bung layout does not match the recipe, score
highest so the operator labels the most informative images first.
"""
from __future__ import annotations

from dataclasses import dataclass

# Weights chosen so a missing battery dominates, per-bung errors accumulate
# linearly, stray bungs matter slightly more than a single miscount, and low
# confidence nudges otherwise-plausible images up the queue.
NO_BATTERY_PENALTY = 10.0
OUTSIDE_BUNG_WEIGHT = 1.5
LOW_CONF_WEIGHT = 2.0


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def disagreement_score(
    battery_count: int,
    per_battery_counts: list[int],
    outside: int,
    expected: int,
    avg_conf: float | None = None,
) -> float:
    """Higher = more disagreement with the recipe = label sooner.

    - No detected battery is the strongest signal that the image needs a human.
    - Each battery contributes the absolute difference between its detected bung
      count and the expected count.
    - Bungs detected outside every battery add uncertainty.
    - Low average detection confidence raises the score so borderline images are
      not assumed correct.
    """
    score = 0.0
    if battery_count <= 0:
        score += NO_BATTERY_PENALTY
    for c in per_battery_counts:
        score += abs(int(c) - int(expected))
    score += OUTSIDE_BUNG_WEIGHT * max(0, int(outside))
    if avg_conf is not None:
        score += LOW_CONF_WEIGHT * (1.0 - _clamp01(float(avg_conf)))
    return score


@dataclass
class QueueItem:
    key: str
    score: float


def rank_items(items: list[QueueItem]) -> list[QueueItem]:
    """Sort highest-disagreement first; ties keep a stable key order."""
    return sorted(items, key=lambda it: (-float(it.score), str(it.key)))

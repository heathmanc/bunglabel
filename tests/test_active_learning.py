"""Unit tests for the pure active-learning scoring (headless-safe)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bung_labeler.core import active_learning as al


def test_perfect_match_scores_zero():
    assert al.disagreement_score(1, [6], 0, 6) == 0.0
    assert al.disagreement_score(2, [6, 6], 0, 6) == 0.0


def test_no_battery_is_high():
    assert al.disagreement_score(0, [], 0, 6) >= al.NO_BATTERY_PENALTY


def test_miscount_accumulates():
    # One battery short by 2, plus one stray bung.
    s = al.disagreement_score(1, [4], 1, 6)
    assert s == 2 + al.OUTSIDE_BUNG_WEIGHT


def test_low_confidence_raises_score():
    high = al.disagreement_score(1, [6], 0, 6, avg_conf=0.95)
    low = al.disagreement_score(1, [6], 0, 6, avg_conf=0.10)
    assert low > high


def test_confidence_clamped():
    # Out-of-range confidence must not produce negative or runaway terms.
    assert al.disagreement_score(1, [6], 0, 6, avg_conf=2.0) == 0.0
    assert al.disagreement_score(1, [6], 0, 6, avg_conf=-1.0) == al.LOW_CONF_WEIGHT


def test_rank_orders_high_first_with_stable_ties():
    items = [
        al.QueueItem("b", 1.0),
        al.QueueItem("a", 5.0),
        al.QueueItem("c", 1.0),
    ]
    ranked = al.rank_items(items)
    assert [it.key for it in ranked] == ["a", "b", "c"]


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    raise SystemExit(1 if failures else 0)

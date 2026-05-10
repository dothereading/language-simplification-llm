"""Tests for the DPO rejected-strategy assignment logic."""
from __future__ import annotations

from collections import Counter

import pytest

distill = pytest.importorskip("langsimp.data.distill", reason="distill.py")


# Common test fixture
def _strats():
    return distill.build_dpo_strategies(
        weak_teacher="weak/model",
        strong_teacher="strong/model",
    )


class TestBuildDpoStrategies:
    def test_includes_four_strategies(self):
        s = _strats()
        names = [x.name for x in s]
        assert "weak-distill" in names
        assert "summarize" in names
        assert "eli5" in names
        assert "clarify" in names

    def test_weights_sum_to_one(self):
        s = _strats()
        total = sum(x.weight for x in s)
        assert total == pytest.approx(1.0)

    def test_weak_distill_uses_weak_teacher(self):
        s = _strats()
        weak = next(x for x in s if x.name == "weak-distill")
        assert weak.model == "weak/model"

    def test_other_strategies_use_strong_teacher(self):
        s = _strats()
        for x in s:
            if x.name != "weak-distill":
                assert x.model == "strong/model"


class TestPickStrategies:
    def test_returns_n_assignments(self):
        s = _strats()
        out = distill.pick_strategies(15, s, seed=0)
        assert len(out) == 15

    def test_deterministic_with_same_seed(self):
        s = _strats()
        a = distill.pick_strategies(15, s, seed=42)
        b = distill.pick_strategies(15, s, seed=42)
        assert [x.name for x in a] == [x.name for x in b]

    def test_different_seeds_produce_different_orders(self):
        s = _strats()
        a = distill.pick_strategies(15, s, seed=1)
        b = distill.pick_strategies(15, s, seed=2)
        assert [x.name for x in a] != [x.name for x in b]

    def test_respects_weights_for_n_15(self):
        # weights 0.60, 0.15, 0.15, 0.10 over 15 records → 9, 2, 2, 2
        s = _strats()
        out = distill.pick_strategies(15, s, seed=0)
        counts = Counter(x.name for x in out)
        assert counts["weak-distill"] == 9
        assert counts["summarize"] == 2
        assert counts["eli5"] == 2
        assert counts["clarify"] == 2

    def test_handles_n_smaller_than_strategies(self):
        # With n=2 we can't represent every strategy; should still return 2
        s = _strats()
        out = distill.pick_strategies(2, s, seed=0)
        assert len(out) == 2

    def test_approximate_ratios_at_scale(self):
        s = _strats()
        out = distill.pick_strategies(200, s, seed=0)
        counts = Counter(x.name for x in out)
        assert counts["weak-distill"] == 120  # exactly 0.60 * 200
        assert counts["summarize"] == 30
        assert counts["eli5"] == 30
        assert counts["clarify"] == 20

    def test_never_negative_counts(self):
        s = _strats()
        # No combination of weights and n should produce negative counts
        for n in [1, 2, 3, 7, 15, 100, 1000]:
            out = distill.pick_strategies(n, s, seed=0)
            assert len(out) == n

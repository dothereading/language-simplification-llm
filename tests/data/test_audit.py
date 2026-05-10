"""Tests for the dataset_audit module — flags low-quality (complex, simple) pairs.

The judge-based check is mocked; the pure-Python checks (length ratio, pacing)
exercise their real logic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import langsimp.data.audit as da
from langsimp.verifier import BaseJudge


class StubJudge(BaseJudge):
    def __init__(self, level: str = "A2"):
        self.level = level

    def evaluate(self, prompt: str):
        return {"level": self.level, "complex_features": "", "reasoning": "", "confidence": 0.9}


def _write_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "data.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


class TestAuditRecord:
    def test_clean_record_has_no_flags(self):
        rec = {"complex": "A short source paragraph.", "simple": "A short rewrite."}
        result = da.audit_record(rec, judge=None)
        assert result["flags"] == []

    def test_flags_excessive_length_inflation(self):
        src = "Short source."
        out = " ".join(["padding"] * 50)  # ratio way past hard cap
        rec = {"complex": src, "simple": out}
        result = da.audit_record(rec, judge=None)
        assert "length_inflated" in result["flags"]

    def test_flags_monotonous_pacing(self):
        # Six sentences all starting "It is" — score 0
        out = " ".join(["It is X."] * 6)
        rec = {"complex": out, "simple": out}  # length-ratio=1, no length flag
        result = da.audit_record(rec, judge=None)
        assert "monotonous" in result["flags"]

    def test_records_pacing_and_length_scores_numerically(self):
        rec = {"complex": "A B C D E F G H I J.", "simple": "A B C."}
        result = da.audit_record(rec, judge=None)
        assert "length_ratio" in result["scores"]
        assert "pacing_variety" in result["scores"]
        assert isinstance(result["scores"]["length_ratio"], float)
        assert isinstance(result["scores"]["pacing_variety"], float)

    def test_judge_level_recorded_when_judge_provided(self):
        rec = {"complex": "src", "simple": "Cats sleep. Dogs run."}
        result = da.audit_record(rec, judge=StubJudge(level="A2"))
        assert result["judge_level"] == "A2"
        assert "too_easy" not in result["flags"]
        assert "too_hard" not in result["flags"]

    def test_flags_too_easy_when_judge_says_a1(self):
        rec = {"complex": "src", "simple": "Cats. Dogs. Birds."}
        result = da.audit_record(rec, judge=StubJudge(level="A1"))
        assert "too_easy" in result["flags"]

    def test_flags_too_hard_when_judge_says_b1(self):
        rec = {"complex": "src", "simple": "Cats sleep."}
        result = da.audit_record(rec, judge=StubJudge(level="B1"))
        assert "too_hard" in result["flags"]


class TestAuditFile:
    def test_aggregates_per_record_results(self, tmp_path):
        rows = [
            {"complex": "src one here.", "simple": "out one here."},
            {"complex": "src two.", "simple": " ".join(["pad"] * 20)},
        ]
        path = _write_jsonl(tmp_path, rows)
        report = da.audit_file(path, judge=None)
        assert len(report["records"]) == 2
        assert "totals" in report
        assert report["totals"]["count"] == 2
        # The padded one should be flagged length_inflated
        flag_counts = report["totals"]["flag_counts"]
        assert flag_counts.get("length_inflated", 0) == 1

    def test_includes_score_distribution_summary(self, tmp_path):
        rows = [{"complex": "x y z", "simple": "x y"} for _ in range(3)]
        path = _write_jsonl(tmp_path, rows)
        report = da.audit_file(path, judge=None)
        assert "mean_length_ratio" in report["totals"]
        assert "mean_pacing_variety" in report["totals"]

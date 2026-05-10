"""Tests for verifier.py — judges, tests, and reward aggregation.

The judge is mocked so no LM Studio process is required.
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from langsimp.verifier import (
    BaseJudge,
    DifficultyRankingTest,
    LocalJudge,
    PacingVarietyTest,
    RewardVerifier,
    _clean,
    length_ratio_score,
    truncate_to_words,
    windowed_excerpts,
)


class StubJudge(BaseJudge):
    """In-memory judge that returns a pre-set response."""
    def __init__(self, response: Dict[str, Any]):
        self.response = response
        self.calls: list[str] = []

    def evaluate(self, prompt: str) -> Dict[str, Any]:
        self.calls.append(prompt)
        return self.response


class TestClean:
    def test_strips_urls(self):
        assert _clean("see https://example.com/foo for more") == "see  for more"

    def test_strips_bracketed_refs(self):
        assert _clean("citation [1] needed [42] here") == "citation  needed  here"


class TestTruncateToWords:
    def test_short_text_returned_whole(self):
        assert truncate_to_words("a b c", n_words=10) == "a b c"

    def test_long_text_truncated_to_n_words(self):
        text = " ".join(str(i) for i in range(50))
        out = truncate_to_words(text, n_words=10)
        assert out.split() == [str(i) for i in range(10)]

    def test_cleans_before_truncation(self):
        text = "see https://example.com/x for context [1] one two three four five"
        out = truncate_to_words(text, n_words=4)
        # URL and ref removed by _clean before split
        assert out == "see for context one"


class TestWindowedExcerpts:
    def test_single_window_when_short(self):
        assert windowed_excerpts("a b c", n_words=10) == ["a b c"]

    def test_three_evenly_spaced_windows(self):
        text = " ".join(str(i) for i in range(30))
        out = windowed_excerpts(text, n_words=10, n_windows=3)
        assert len(out) == 3
        assert out[0].split() == [str(i) for i in range(10)]
        assert out[-1].split() == [str(i) for i in range(20, 30)]

    def test_dedups_when_text_too_short_for_distinct_windows(self):
        text = " ".join(str(i) for i in range(11))
        out = windowed_excerpts(text, n_words=10, n_windows=3)
        # Only two distinct starting positions exist (0 and 1)
        assert len(out) <= 3
        assert len(set(out)) == len(out)


class TestLocalJudgeAuth:
    """LocalJudge against a local LM Studio endpoint sends no auth header.
    When `api_key` is set (e.g. for OpenRouter), it must send
    `Authorization: Bearer <key>` so the request authenticates."""

    def _mock_post(self, monkeypatch):
        captured: dict = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            resp = MagicMock()
            resp.json.return_value = {
                "choices": [{"message": {"content": '{"ok": 1}'}}]
            }
            resp.raise_for_status.return_value = None
            return resp

        monkeypatch.setattr("langsimp.verifier.requests.post", fake_post)
        return captured

    def test_no_auth_header_when_no_api_key(self, monkeypatch):
        captured = self._mock_post(monkeypatch)
        j = LocalJudge(base_url="http://x", model_name="m")
        j.evaluate("hello")
        # Either headers absent / None, or Authorization not present.
        headers = captured.get("headers") or {}
        assert "Authorization" not in headers

    def test_sends_bearer_when_api_key_set(self, monkeypatch):
        captured = self._mock_post(monkeypatch)
        j = LocalJudge(base_url="https://openrouter.ai/api/v1",
                       model_name="anthropic/claude-haiku-latest",
                       api_key="sk-test-123")
        j.evaluate("hello")
        headers = captured.get("headers") or {}
        assert headers.get("Authorization") == "Bearer sk-test-123"

    def test_forwards_model_alias(self, monkeypatch):
        captured = self._mock_post(monkeypatch)
        j = LocalJudge(base_url="https://openrouter.ai/api/v1",
                       model_name="anthropic/claude-haiku-latest",
                       api_key="sk-test-123")
        j.evaluate("hello")
        assert captured["json"]["model"] == "anthropic/claude-haiku-latest"


class TestLocalJudgeJsonParse:
    def setup_method(self):
        self.j = LocalJudge(base_url="http://x", model_name="m")

    def test_parses_plain_json(self):
        assert self.j._parse_json('{"level": "A2"}') == {"level": "A2"}

    def test_parses_json_in_fenced_block(self):
        raw = 'preamble\n```json\n{"level": "A1"}\n```\ntrailing'
        assert self.j._parse_json(raw) == {"level": "A1"}

    def test_parses_json_in_unlabeled_fence(self):
        raw = '```\n{"level": "B1"}\n```'
        assert self.j._parse_json(raw) == {"level": "B1"}

    def test_falls_back_to_first_brace_block(self):
        raw = 'noise before {"level": "A2", "x": 1} noise after'
        assert self.j._parse_json(raw) == {"level": "A2", "x": 1}


class TestDifficultyRanking:
    def _test_obj(self, **kwargs):
        return DifficultyRankingTest(
            a1_samples=["short"], b1_samples=["long"], a2_samples=["med"], **kwargs,
        )

    def test_returns_1_when_judge_says_a2(self):
        t = self._test_obj()
        score = t.run("some text", StubJudge({"level": "A2"}))
        assert score == 1.0

    def test_returns_0_when_judge_says_b1(self):
        t = self._test_obj()
        assert t.run("some text", StubJudge({"level": "B1"})) == 0.0

    def test_normalizes_b2_variants_to_b2plus(self):
        t = self._test_obj()
        for label in ["B2", "C1", "C2"]:
            assert t.run("x", StubJudge({"level": label})) == 0.0

    def test_normalizes_below_a1_variants(self):
        t = self._test_obj()
        for label in ["BELOW A1", "PRE-A1", "<<A1"]:
            assert t.run("x", StubJudge({"level": label})) == 0.0

    def test_returns_0_for_unrecognized_level(self):
        t = self._test_obj()
        assert t.run("x", StubJudge({"level": "QQQ"})) == 0.0

    def test_prompt_includes_candidate_text(self):
        t = self._test_obj()
        judge = StubJudge({"level": "A2"})
        t.run("CANDIDATE_MARKER text here", judge)
        assert "CANDIDATE_MARKER" in judge.calls[0]


class TestDifficultyRankingClassify:
    """`classify()` returns the level *label* — the eval harness needs this
    information (not just the binary 1.0/0.0 from run())."""

    def _t(self):
        return DifficultyRankingTest(
            a1_samples=["short"], b1_samples=["long"], a2_samples=["med"],
        )

    def test_returns_level_string(self):
        t = self._t()
        assert t.classify("text", StubJudge({"level": "A2"})) == "A2"
        assert t.classify("text", StubJudge({"level": "B1"})) == "B1"
        assert t.classify("text", StubJudge({"level": "A1"})) == "A1"

    def test_normalizes_b2_variants_to_b2plus(self):
        t = self._t()
        for lab in ["B2", "C1", "C2", "C1+"]:
            assert t.classify("x", StubJudge({"level": lab})) == "B2+"

    def test_normalizes_below_a1_variants(self):
        t = self._t()
        for lab in ["BELOW A1", "PRE-A1", "<<A1"]:
            assert t.classify("x", StubJudge({"level": lab})) == "<A1"

    def test_unrecognized_returns_NA(self):
        t = self._t()
        assert t.classify("x", StubJudge({"level": "QQQ"})) == "NA"
        assert t.classify("x", StubJudge({})) == "NA"

    def test_run_and_classify_agree(self):
        # Whatever level classify returns, run scores 1.0 iff that level is A2
        t = self._t()
        for lab in ["A2", "A1", "B1", "B2", "QQQ"]:
            judge = StubJudge({"level": lab})
            classified = t.classify("x", judge)
            judge2 = StubJudge({"level": lab})
            scored = t.run("x", judge2)
            assert (scored == 1.0) == (classified == "A2")


class TestPacingVariety:
    def setup_method(self):
        self.t = PacingVarietyTest()

    def test_all_same_opening_scores_zero(self):
        # Four sentences starting "It is" → no variety
        text = "It is red. It is blue. It is green. It is gold."
        assert self.t.run(text, judge=None) == pytest.approx(0.0)

    def test_all_distinct_openings_score_high(self):
        text = "Bob ran fast. The cat slept. Snow fell hard. Cars passed by."
        # 4 unique openings → 1 - 4*(1/4)^2 = 0.75
        assert self.t.run(text, judge=None) == pytest.approx(0.75)

    def test_partial_repetition(self):
        # 3x "It is" and 1x "Snow fell" → 1 - ((3/4)^2 + (1/4)^2) = 1 - 0.625 = 0.375
        text = "It is hot. It is bright. It is loud. Snow fell down."
        assert self.t.run(text, judge=None) == pytest.approx(0.375)

    def test_single_sentence_returns_one(self):
        assert self.t.run("Just one sentence here.", judge=None) == 1.0

    def test_empty_text_returns_one(self):
        # No sentences to be monotonous about; don't penalize.
        assert self.t.run("", judge=None) == 1.0

    def test_ignores_internal_punctuation(self):
        # Comma-separated clauses inside a single sentence don't count as separate sentences.
        text = "Bob ran, jumped, and slept. Cats prowl at night."
        assert self.t.run(text, judge=None) == pytest.approx(0.5)


class TestLengthRatioScore:
    def test_output_shorter_than_source_scores_one(self):
        assert length_ratio_score("a b c d e f", "a b c") == 1.0

    def test_output_equal_length_scores_one(self):
        assert length_ratio_score("a b c", "x y z") == 1.0

    def test_within_soft_cap_scores_one(self):
        # Default soft_cap=1.3; output 1.2x source → no penalty
        src = " ".join(["w"] * 10)
        out = " ".join(["w"] * 12)
        assert length_ratio_score(src, out) == 1.0

    def test_at_hard_cap_scores_zero(self):
        src = " ".join(["w"] * 10)
        out = " ".join(["w"] * 20)  # 2x → hard cap default
        assert length_ratio_score(src, out) == pytest.approx(0.0)

    def test_between_soft_and_hard_cap_decays_linearly(self):
        # Default soft=1.3, hard=2.0. Ratio 1.65 is halfway → score 0.5
        src = " ".join(["w"] * 100)
        out = " ".join(["w"] * 165)
        assert length_ratio_score(src, out) == pytest.approx(0.5, abs=0.01)

    def test_past_hard_cap_clamps_to_zero(self):
        src = " ".join(["w"] * 10)
        out = " ".join(["w"] * 50)  # 5x
        assert length_ratio_score(src, out) == 0.0

    def test_empty_source_returns_zero(self):
        assert length_ratio_score("", "anything") == 0.0

    def test_custom_caps(self):
        src = " ".join(["w"] * 10)
        out = " ".join(["w"] * 11)  # 1.1x
        # With soft=1.0, hard=1.2, ratio 1.1 is halfway → 0.5
        assert length_ratio_score(src, out, soft_cap=1.0, hard_cap=1.2) == pytest.approx(0.5, abs=0.01)


class TestRewardVerifier:
    def test_no_tests_returns_zero(self):
        v = RewardVerifier(judge=StubJudge({}))
        assert v.verify("anything") == 0.0

    def test_averages_test_scores(self):
        class FixedScore:
            def __init__(self, s): self.s = s
            def run(self, text, judge): return self.s

        v = RewardVerifier(judge=StubJudge({}))
        v.add_test(FixedScore(1.0))
        v.add_test(FixedScore(0.0))
        assert v.verify("x") == 0.5

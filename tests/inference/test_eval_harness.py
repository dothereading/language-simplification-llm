"""Tests for the held-out eval harness.

The model and judge are mocked; we exercise the orchestration logic
(prompt building, per-record evaluation, aggregate summarization).
"""
from __future__ import annotations

import json

import pytest

eval_mod = pytest.importorskip("langsimp.inference.eval_harness", reason="eval_harness.py not implemented yet (RED)")


class TestEvaluateEvalSet:
    def test_runs_each_record_through_generate_and_classify(self):
        records = [
            {"complex": "src1", "title": "t1"},
            {"complex": "src2", "title": "t2"},
        ]
        gen = lambda x: f"simplified({x})"
        cls = lambda x: "A2"
        results = eval_mod.evaluate_eval_set(records, gen, cls)
        assert len(results) == 2
        assert results[0]["complex"] == "src1"
        assert results[0]["output"] == "simplified(src1)"
        assert results[0]["level"] == "A2"
        assert results[0]["title"] == "t1"

    def test_records_word_counts(self):
        records = [{"complex": "a b c d e"}]
        gen = lambda x: "x y z w"
        cls = lambda x: "A2"
        results = eval_mod.evaluate_eval_set(records, gen, cls)
        assert results[0]["source_words"] == 5
        assert results[0]["output_words"] == 4

    def test_handles_classify_failure_with_NA(self):
        # If classify raises (judge unavailable), record level as "NA" rather than crash
        records = [{"complex": "src"}]
        gen = lambda x: "out"
        def cls(x): raise RuntimeError("judge down")
        results = eval_mod.evaluate_eval_set(records, gen, cls)
        assert results[0]["level"] == "NA"


class TestSummarize:
    def test_level_distribution(self):
        results = [
            {"level": "A2", "source_words": 50, "output_words": 60},
            {"level": "A2", "source_words": 50, "output_words": 60},
            {"level": "A1", "source_words": 50, "output_words": 30},
            {"level": "B1", "source_words": 50, "output_words": 80},
        ]
        s = eval_mod.summarize(results)
        assert s["count"] == 4
        assert s["level_counts"] == {"A2": 2, "A1": 1, "B1": 1}
        assert s["pct_a2"] == pytest.approx(0.5)
        assert s["pct_too_easy"] == pytest.approx(0.25)  # A1 + <A1
        assert s["pct_too_hard"] == pytest.approx(0.25)  # B1 + B2+

    def test_too_hard_includes_b2plus(self):
        results = [
            {"level": "A2", "source_words": 50, "output_words": 60},
            {"level": "B2+", "source_words": 50, "output_words": 60},
        ]
        s = eval_mod.summarize(results)
        assert s["pct_too_hard"] == pytest.approx(0.5)

    def test_too_easy_includes_below_a1(self):
        results = [
            {"level": "A2", "source_words": 50, "output_words": 60},
            {"level": "<A1", "source_words": 50, "output_words": 60},
        ]
        s = eval_mod.summarize(results)
        assert s["pct_too_easy"] == pytest.approx(0.5)

    def test_NA_doesnt_count_toward_a2_or_failures(self):
        results = [
            {"level": "A2", "source_words": 50, "output_words": 60},
            {"level": "NA", "source_words": 50, "output_words": 60},
        ]
        s = eval_mod.summarize(results)
        # NA is its own bucket; not A2, not too_easy, not too_hard
        assert s["pct_a2"] == pytest.approx(0.5)
        assert s["pct_too_easy"] == 0.0
        assert s["pct_too_hard"] == 0.0

    def test_length_stats(self):
        results = [
            {"level": "A2", "source_words": 100, "output_words": 110},
            {"level": "A2", "source_words": 100, "output_words": 130},
        ]
        s = eval_mod.summarize(results)
        assert s["mean_length_ratio"] == pytest.approx(1.2)

    def test_handles_empty_results(self):
        s = eval_mod.summarize([])
        assert s["count"] == 0
        assert s["pct_a2"] == 0.0


class TestCleanGeneration:
    """mlx-lm doesn't always stop at chat-template turn markers — Gemma in
    particular emits <end_of_turn> and then keeps generating garbage until
    max_tokens. We post-process to strip everything from the first stop
    token onward."""

    def test_truncates_at_end_of_turn(self):
        raw = "Real output here.<end_of_turn>noise noise noise"
        assert eval_mod.clean_generation(raw) == "Real output here."

    def test_truncates_at_eos(self):
        raw = "Real output.<eos>more noise"
        assert eval_mod.clean_generation(raw) == "Real output."

    def test_truncates_at_im_end(self):
        raw = "Output.<|im_end|>tail"
        assert eval_mod.clean_generation(raw) == "Output."

    def test_handles_repeated_markers(self):
        raw = "Good text.<end_of_turn><end_of_turn><end_of_turn>"
        assert eval_mod.clean_generation(raw) == "Good text."

    def test_strips_whitespace(self):
        assert eval_mod.clean_generation("  hello  \n") == "hello"

    def test_no_marker_returns_full_text(self):
        assert eval_mod.clean_generation("Plain output here.") == "Plain output here."

    def test_takes_earliest_marker(self):
        raw = "Text<end_of_turn>middle<eos>tail"
        assert eval_mod.clean_generation(raw) == "Text"


class TestBuildEvalPrompt:
    def test_uses_sft_system_prompt(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return json.dumps(messages)
        prompt = eval_mod.build_eval_prompt("complex paragraph", FakeTokenizer())
        msgs = json.loads(prompt)
        roles = [m["role"] for m in msgs]
        assert "system" in roles
        assert any(m["role"] == "user" and m["content"] == "complex paragraph" for m in msgs)

    def test_passes_add_generation_prompt(self):
        captured = {}
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                captured["add_generation_prompt"] = add_generation_prompt
                captured["tokenize"] = tokenize
                return ""
        eval_mod.build_eval_prompt("x", FakeTokenizer())
        # The model needs the prompt suffix that begins assistant turn.
        assert captured["add_generation_prompt"] is True
        assert captured["tokenize"] is False

"""Tests for dataset record formatting and split logic.

Currently bridges between the OLD `prepare_mlx_data` module and the NEW
`datasets` module that the refactor will introduce. The shared behaviors:
  * `to_mlx_sft_record(complex, simple)` → MLX chat record
  * `to_mlx_dpo_record(prompt, chosen, rejected)` → MLX DPO record
  * `split_train_valid(rows, valid_frac, seed)` → (train, valid) lists
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _import_target():
    import langsimp.data.mlx_format as ds
    return ds, "new"


mod, flavor = _import_target()


class TestSftRecord:
    def test_sft_record_shape(self):
        if flavor == "new":
            rec = mod.to_mlx_sft_record("complex text", "simple text")
        else:
            rec = mod.to_record("complex text", "simple text")
        assert "messages" in rec
        roles = [m["role"] for m in rec["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert rec["messages"][1]["content"] == "complex text"
        assert rec["messages"][2]["content"] == "simple text"

    def test_strips_whitespace_in_content(self):
        if flavor == "new":
            rec = mod.to_mlx_sft_record("  complex  ", "  simple  ")
        else:
            rec = mod.to_record("  complex  ", "  simple  ")
        assert rec["messages"][1]["content"] == "complex"
        assert rec["messages"][2]["content"] == "simple"


@pytest.mark.skipif(flavor == "old", reason="DPO record helper only exists in new datasets module")
class TestDpoRecord:
    def test_dpo_record_shape(self):
        rec = mod.to_mlx_dpo_record("prompt", "chosen", "rejected")
        assert set(rec.keys()) >= {"system", "prompt", "chosen", "rejected"}
        assert rec["prompt"] == "prompt"
        assert rec["chosen"] == "chosen"
        assert rec["rejected"] == "rejected"
        assert isinstance(rec["system"], str) and rec["system"]


@pytest.mark.skipif(flavor == "old", reason="hash-based split only exists in new datasets module")
class TestAssignSplit:
    def test_deterministic_for_same_input(self):
        assert mod.assign_split("para A", 0.1, seed=42) == mod.assign_split("para A", 0.1, seed=42)

    def test_returns_only_train_or_valid(self):
        for i in range(20):
            assert mod.assign_split(f"x{i}", 0.1, seed=0) in ("train", "valid")

    def test_approximate_valid_fraction(self):
        results = [mod.assign_split(f"prompt_{i}", 0.1, seed=0) for i in range(2000)]
        valid_frac = results.count("valid") / 2000
        # Hash-based, expect within ±2pp of target with N=2000
        assert 0.08 < valid_frac < 0.12

    def test_different_seeds_produce_different_assignments(self):
        diff = sum(
            mod.assign_split(f"k{i}", 0.5, seed=1) != mod.assign_split(f"k{i}", 0.5, seed=2)
            for i in range(200)
        )
        assert diff > 50  # well above zero, well below all 200


@pytest.mark.skipif(flavor == "old", reason="select_eval_keys only exists in new datasets module")
class TestSelectEvalKeys:
    def test_picks_n_keys(self):
        keys = [f"k{i}" for i in range(100)]
        out = mod.select_eval_keys(keys, n=10, seed=0)
        assert len(out) == 10
        assert set(out) <= set(keys)

    def test_deterministic(self):
        keys = [f"k{i}" for i in range(50)]
        a = mod.select_eval_keys(keys, n=5, seed=0)
        b = mod.select_eval_keys(keys, n=5, seed=0)
        assert a == b

    def test_stable_when_dataset_grows(self):
        # The whole point of hash-based selection: adding more keys later
        # shouldn't change which keys are eval keys.
        small = [f"k{i}" for i in range(50)]
        big = small + [f"new_{i}" for i in range(50)]
        eval_small = set(mod.select_eval_keys(small, n=5, seed=0))
        eval_big = set(mod.select_eval_keys(big, n=5, seed=0))
        # Every old eval key should remain an eval key (or be displaced by a new
        # key that hashes lower — but at most a few should change).
        # Strict version: keys present in both eval sets that come from `small`
        # should be a strict subset relationship — but allow a couple of
        # displacements when the new candidate keys hash lower.
        retained_from_small = eval_big & set(small)
        # At most a few of the original 5 should have been displaced.
        assert len(retained_from_small) >= 3

    def test_seed_changes_selection(self):
        keys = [f"k{i}" for i in range(50)]
        a = set(mod.select_eval_keys(keys, n=10, seed=1))
        b = set(mod.select_eval_keys(keys, n=10, seed=2))
        assert a != b


@pytest.mark.skipif(flavor == "old", reason="carve_eval only exists in new datasets module")
class TestCarveEval:
    def _write(self, tmp_path, n):
        src = tmp_path / "sft.jsonl"
        with open(src, "w") as f:
            for i in range(n):
                f.write(json.dumps({
                    "complex": f"para {i}", "simple": f"easy {i}", "title": f"t{i}"
                }) + "\n")
        return src

    def test_writes_n_records(self, tmp_path):
        src = self._write(tmp_path, 20)
        out = tmp_path / "eval.jsonl"
        n = mod.carve_eval(src, out, n=5, seed=0)
        assert n == 5
        with open(out) as f:
            recs = [json.loads(l) for l in f if l.strip()]
        assert len(recs) == 5
        assert all("complex" in r for r in recs)

    def test_does_not_modify_source(self, tmp_path):
        src = self._write(tmp_path, 10)
        original = src.read_text()
        mod.carve_eval(src, tmp_path / "eval.jsonl", n=3, seed=0)
        assert src.read_text() == original

    def test_idempotent_with_same_seed(self, tmp_path):
        src = self._write(tmp_path, 20)
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        mod.carve_eval(src, a, n=5, seed=42)
        mod.carve_eval(src, b, n=5, seed=42)
        assert a.read_text() == b.read_text()

    def test_n_larger_than_source_returns_all(self, tmp_path):
        src = self._write(tmp_path, 5)
        n = mod.carve_eval(src, tmp_path / "out.jsonl", n=100, seed=0)
        assert n == 5


@pytest.mark.skipif(flavor == "old", reason="load_eval_prompts only exists in new datasets module")
class TestLoadEvalPrompts:
    def test_returns_set_of_complex_field(self, tmp_path):
        p = tmp_path / "eval.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps({"complex": "foo", "simple": "f"}) + "\n")
            f.write(json.dumps({"complex": "bar", "simple": "b"}) + "\n")
        assert mod.load_eval_prompts(p) == {"foo", "bar"}

    def test_missing_file_returns_empty_set(self, tmp_path):
        assert mod.load_eval_prompts(tmp_path / "nope.jsonl") == set()

    def test_handles_dpo_schema_too(self, tmp_path):
        # If we ever point at a DPO-shaped file, the prompt field is "prompt"
        p = tmp_path / "eval.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps({"prompt": "foo", "chosen": "c", "rejected": "r"}) + "\n")
        assert mod.load_eval_prompts(p) == {"foo"}


@pytest.mark.skipif(flavor == "old", reason="prepare_sft_splits only exists in new datasets module")
class TestPrepareSftSplits:
    def test_excludes_eval_prompts(self):
        rows = [{"complex": f"p{i}", "simple": f"s{i}"} for i in range(10)]
        train, valid = mod.prepare_sft_splits(rows, eval_prompts={"p3", "p7"}, valid_frac=0.2, seed=0)
        prompts = {r["complex"] for r in train + valid}
        assert "p3" not in prompts and "p7" not in prompts
        assert len(train) + len(valid) == 8

    def test_no_eval_keeps_all(self):
        rows = [{"complex": f"p{i}", "simple": ""} for i in range(20)]
        train, valid = mod.prepare_sft_splits(rows, eval_prompts=set(), valid_frac=0.1, seed=0)
        assert len(train) + len(valid) == 20


@pytest.mark.skipif(flavor == "old", reason="prepare_dpo_splits only exists in new datasets module")
class TestPrepareDpoSplits:
    def test_excludes_eval_prompts(self):
        rows = [{"prompt": f"p{i}", "chosen": "c", "rejected": "r"} for i in range(10)]
        train, valid = mod.prepare_dpo_splits(rows, eval_prompts={"p2", "p5"}, valid_frac=0.2, seed=0)
        prompts = {r["prompt"] for r in train + valid}
        assert "p2" not in prompts and "p5" not in prompts


@pytest.mark.skipif(flavor == "old", reason="GRPO format only exists in new datasets module")
class TestToMlxGrpoRecord:
    def test_record_shape(self):
        rec = mod.to_mlx_grpo_record("complex source", "opus reference")
        # mlx_lm_lora GRPO expects {prompt, answer, system?}
        assert "prompt" in rec
        assert "answer" in rec
        assert "system" in rec
        assert rec["prompt"] == "complex source"
        assert rec["answer"] == "opus reference"
        assert isinstance(rec["system"], str) and rec["system"]


@pytest.mark.skipif(flavor == "old", reason="GRPO valid carve only exists in new datasets module")
class TestPrepareGrpoSplits:
    def test_excludes_eval_prompts(self):
        rows = [{"complex": f"p{i}", "simple": f"s{i}"} for i in range(20)]
        train, valid = mod.prepare_grpo_splits(
            rows, eval_prompts={"p3", "p7"}, valid_n=4, seed=0,
        )
        prompts = {r["complex"] for r in train + valid}
        assert "p3" not in prompts and "p7" not in prompts
        assert len(train) + len(valid) == 18

    def test_holds_out_n_records_for_valid(self):
        rows = [{"complex": f"p{i}", "simple": f"s{i}"} for i in range(50)]
        train, valid = mod.prepare_grpo_splits(rows, eval_prompts=set(), valid_n=10, seed=0)
        assert len(valid) == 10
        assert len(train) == 40

    def test_train_sorted_by_source_length_for_curriculum(self):
        # train should be sorted ascending by source word count so easier
        # (shorter) prompts come first during GRPO.
        rows = [
            {"complex": " ".join(["w"] * n), "simple": ""} for n in [50, 200, 100, 30, 150]
        ]
        train, _ = mod.prepare_grpo_splits(rows, eval_prompts=set(), valid_n=1, seed=0)
        lengths = [len(r["complex"].split()) for r in train]
        assert lengths == sorted(lengths)

    def test_deterministic_with_seed(self):
        rows = [{"complex": f"p{i}", "simple": ""} for i in range(40)]
        a_train, a_valid = mod.prepare_grpo_splits(rows, eval_prompts=set(), valid_n=5, seed=42)
        b_train, b_valid = mod.prepare_grpo_splits(rows, eval_prompts=set(), valid_n=5, seed=42)
        assert {r["complex"] for r in a_valid} == {r["complex"] for r in b_valid}


@pytest.mark.skipif(flavor == "old", reason="hash-based splits only exist in new datasets module")
class TestSftDpoSplitConsistency:
    def test_same_prompt_lands_in_same_split(self):
        # The whole point of hash-based assignment: a prompt that lands in
        # SFT-valid lands in DPO-valid too, so the validation prompts are
        # consistent across stages.
        prompts = [f"para_{i}" for i in range(50)]
        sft_rows = [{"complex": p, "simple": "x"} for p in prompts]
        dpo_rows = [{"prompt": p, "chosen": "c", "rejected": "r"} for p in prompts]
        sft_t, sft_v = mod.prepare_sft_splits(sft_rows, eval_prompts=set(), valid_frac=0.2, seed=0)
        dpo_t, dpo_v = mod.prepare_dpo_splits(dpo_rows, eval_prompts=set(), valid_frac=0.2, seed=0)
        assert {r["complex"] for r in sft_v} == {r["prompt"] for r in dpo_v}
        assert {r["complex"] for r in sft_t} == {r["prompt"] for r in dpo_t}


@pytest.mark.skipif(flavor == "old", reason="split helper only exists in new datasets module")
class TestSplitTrainValid:
    def test_valid_size_respects_fraction(self):
        rows = [{"i": i} for i in range(100)]
        train, valid = mod.split_train_valid(rows, valid_frac=0.1, seed=0)
        assert len(valid) == 10
        assert len(train) == 90

    def test_at_least_one_valid_row(self):
        rows = [{"i": i} for i in range(3)]
        train, valid = mod.split_train_valid(rows, valid_frac=0.0, seed=0)
        assert len(valid) >= 1

    def test_seed_reproducible(self):
        rows = [{"i": i} for i in range(50)]
        a = mod.split_train_valid(rows, valid_frac=0.2, seed=42)
        b = mod.split_train_valid(rows, valid_frac=0.2, seed=42)
        assert a == b

    def test_train_and_valid_disjoint(self):
        rows = [{"i": i} for i in range(50)]
        train, valid = mod.split_train_valid(rows, valid_frac=0.2, seed=1)
        train_ids = {r["i"] for r in train}
        valid_ids = {r["i"] for r in valid}
        assert not (train_ids & valid_ids)
        assert train_ids | valid_ids == set(range(50))

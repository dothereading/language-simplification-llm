"""Tests for the training-log parsers used to forward metrics to W&B.

The parsers are pure regex over real log lines; we test them against the
exact formats emitted by mlx-lm and mlx-lm-lora (sampled from logs/).
"""
from __future__ import annotations

import pytest

train = pytest.importorskip("langsimp.training.runner", reason="train.py not implemented yet")


class TestParseSftLine:
    def test_train_line(self):
        line = "Iter 10: Train loss 2.249, Learning Rate 1.000e-04, It/sec 1.380, Tokens/sec 404.447, Trained Tokens 2931, Peak mem 4.005 GB"
        m = train.parse_sft_line(line)
        assert m == {
            "iter": 10,
            "train/loss": 2.249,
            "train/lr": 1.000e-04,
            "train/it_per_sec": 1.380,
            "train/tok_per_sec": 404.447,
            "train/trained_tokens": 2931,
            "train/peak_mem_gb": 4.005,
        }

    def test_val_line(self):
        line = "Iter 50: Val loss 1.728, Val took 0.282s"
        m = train.parse_sft_line(line)
        assert m == {"iter": 50, "valid/loss": 1.728}

    def test_unrelated_line_returns_none(self):
        assert train.parse_sft_line("Loading model...") is None
        assert train.parse_sft_line("") is None

    def test_iter_1_val_works(self):
        # The very first val happens at iter 1 in the SFT log
        line = "Iter 1: Val loss 3.903, Val took 2.386s"
        assert train.parse_sft_line(line) == {"iter": 1, "valid/loss": 3.903}


class TestParseDpoLine:
    def test_train_line(self):
        line = "Iter 10: loss 0.011, chosen_r 83.361, rejected_r 65.698, acc 1.000, margin 17.663, lr 5.000e-06, it/s 2.012, tok/s 1172.116, peak_mem 8.719GB"
        m = train.parse_dpo_line(line)
        assert m == {
            "iter": 10,
            "train/loss": 0.011,
            "train/chosen_reward": 83.361,
            "train/rejected_reward": 65.698,
            "train/accuracy": 1.000,
            "train/margin": 17.663,
            "train/lr": 5.000e-06,
            "train/it_per_sec": 2.012,
            "train/tok_per_sec": 1172.116,
            "train/peak_mem_gb": 8.719,
        }

    def test_val_line(self):
        line = "Iter 50: Val loss 0.000, Val chosen reward 0.143, Val rejected reward 0.122, Val accuracy 1.000, Val margin 10.016, Val took 0.765s"
        m = train.parse_dpo_line(line)
        assert m == {
            "iter": 50,
            "valid/loss": 0.000,
            "valid/chosen_reward": 0.143,
            "valid/rejected_reward": 0.122,
            "valid/accuracy": 1.000,
            "valid/margin": 10.016,
        }

    def test_handles_negative_margin(self):
        line = "Iter 5: loss 0.500, chosen_r 10.000, rejected_r 12.000, acc 0.500, margin -2.000, lr 5.000e-06, it/s 2.000, tok/s 1000.000, peak_mem 8.000GB"
        m = train.parse_dpo_line(line)
        assert m["train/margin"] == -2.000

    def test_unrelated_line_returns_none(self):
        assert train.parse_dpo_line("De-quantizing model") is None
        assert train.parse_dpo_line("") is None


class TestParseGrpoLine:
    """GRPO logs in multi-line blocks. The parser is stateful — `Iter N:`
    sets the current iter, subsequent metric lines emit values keyed to that
    iter."""

    def setup_method(self):
        self.p = train.GrpoLogParser()

    def test_iter_header_alone_emits_nothing(self):
        assert self.p("Iter 100:") is None
        assert self.p("Iter 100:\n") is None

    def test_loss_after_header(self):
        self.p("Iter 100:")
        out = self.p("Loss: 0.234")
        assert out == {"iter": 100, "train/loss": 0.234}

    def test_total_rewards_line(self):
        self.p("Iter 50:")
        out = self.p("Total Rewards:  μ=0.612, σ=0.143")
        assert out == {
            "iter": 50,
            "train/reward_mean": 0.612,
            "train/reward_std": 0.143,
        }

    def test_group_rewards_line(self):
        self.p("Iter 50:")
        out = self.p("Group Rewards:  μ=0.500, σ=0.092")
        assert out == {
            "iter": 50,
            "train/group_reward_mean": 0.500,
            "train/group_reward_std": 0.092,
        }

    def test_kl_divergence(self):
        self.p("Iter 50:")
        out = self.p("KL Divergence: 0.000004567890")
        assert out == {"iter": 50, "train/kl": 0.000004567890}

    def test_avg_tokens(self):
        self.p("Iter 50:")
        out = self.p("  • Avg tokens: 84.3")
        assert out == {"iter": 50, "train/avg_tokens": 84.3}

    def test_individual_reward_function(self):
        self.p("Iter 50:")
        out = self.p("  • length: μ=0.910, σ=0.045, cov=92.50%")
        assert out == {
            "iter": 50,
            "train/reward_length_mean": 0.910,
            "train/reward_length_std": 0.045,
        }

    def test_learning_rate(self):
        self.p("Iter 50:")
        out = self.p("Learning Rate: 1.0000e-06")
        assert out == {"iter": 50, "train/lr": 1.0e-06}

    def test_speed(self):
        self.p("Iter 50:")
        out = self.p("Speed: 0.345 it/s, 1234.5 tok/s")
        assert out == {"iter": 50, "train/it_per_sec": 0.345, "train/tok_per_sec": 1234.5}

    def test_memory(self):
        self.p("Iter 50:")
        out = self.p("Memory: 4.987GB")
        assert out == {"iter": 50, "train/peak_mem_gb": 4.987}

    def test_val_loss_single_line(self):
        # Val line is single-line; parser handles it without needing prior Iter:
        out = self.p("Iter 50: Val loss 0.123, Val took 1.234s")
        assert out == {"iter": 50, "valid/loss": 0.123}

    def test_unrelated_line_returns_none(self):
        assert self.p("================================================================================") is None
        assert self.p("") is None
        assert self.p("Generation Stats:") is None


class TestGitShortSha:
    def test_returns_string_when_in_git_repo(self):
        # We're running pytest from the repo root, so this should always work
        sha = train.git_short_sha()
        assert sha is None or (isinstance(sha, str) and len(sha) >= 4)

    def test_returns_none_when_not_in_git_repo(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert train.git_short_sha() is None


class TestDatasetHash:
    def test_hashes_file_contents(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "train.jsonl").write_text('{"a": 1}\n')
        (d / "valid.jsonl").write_text('{"b": 2}\n')
        h1 = train.dataset_hash(d)
        h2 = train.dataset_hash(d)
        assert h1 == h2  # deterministic
        assert isinstance(h1, str) and len(h1) >= 8

    def test_changes_when_content_changes(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "train.jsonl").write_text('{"a": 1}\n')
        (d / "valid.jsonl").write_text('{"b": 2}\n')
        h1 = train.dataset_hash(d)
        (d / "train.jsonl").write_text('{"a": 2}\n')
        h2 = train.dataset_hash(d)
        assert h1 != h2

    def test_handles_missing_dir(self, tmp_path):
        # Missing dir should not crash; return a sentinel.
        h = train.dataset_hash(tmp_path / "nope")
        assert h == "missing"


class TestMakeAdapterDir:
    def test_includes_stage_timestamp_sha(self, tmp_path):
        d = train.make_adapter_dir(tmp_path, "sft", "20260101T120000", "abc1234")
        assert d.parent == tmp_path / "sft"
        assert d.name == "20260101T120000-abc1234"

    def test_creates_directory(self, tmp_path):
        d = train.make_adapter_dir(tmp_path, "dpo", "20260101T120000", "abc1234")
        assert d.exists()
        assert d.is_dir()

    def test_handles_no_sha(self, tmp_path):
        d = train.make_adapter_dir(tmp_path, "sft", "20260101T120000", None)
        assert "nosha" in d.name


class TestUpdateLatestSymlink:
    def test_creates_symlink(self, tmp_path):
        target = tmp_path / "20260101T120000-abc"
        target.mkdir()
        link = tmp_path / "latest"
        train.update_latest_symlink(link, target)
        assert link.is_symlink()
        assert link.resolve() == target.resolve()

    def test_replaces_existing_symlink(self, tmp_path):
        old_target = tmp_path / "old"
        old_target.mkdir()
        new_target = tmp_path / "new"
        new_target.mkdir()
        link = tmp_path / "latest"
        train.update_latest_symlink(link, old_target)
        train.update_latest_symlink(link, new_target)
        assert link.resolve() == new_target.resolve()


class TestBuildRunName:
    def test_includes_stage_and_model(self):
        name = train.build_run_name(stage="sft", model="mlx-community/gemma-3-1b-it-bf16",
                                    config={"iters": 300, "lr": 1e-4, "batch_size": 1})
        assert "sft" in name
        assert "gemma-3-1b" in name

    def test_includes_iters_and_lr(self):
        name = train.build_run_name(stage="dpo", model="mlx-community/gemma-3-1b-it-bf16",
                                    config={"iters": 500, "lr": 5e-6, "beta": 0.1})
        assert "iters500" in name
        assert "lr5e-06" in name or "lr5.0e-06" in name or "5e-06" in name

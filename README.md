![alt text](simple_lm_banner.jpg)

# Language Simplification Fine-Tuning

Fine-tune a small Gemma model to rewrite complex English at **CEFR A2** (Elementary) level. See [`PLAN.md`](PLAN.md) for the staged pipeline (data → SFT → DPO → GRPO → eval) and [`CLAUDE.md`](CLAUDE.md) for the development conventions used in this repo.

## Layout

The code lives in a single `langsimp/` package with subpackages organized by concern (data, training, inference) plus a shared `verifier` module. Each subpackage's CLIs are invoked via `python -m langsimp.<subpkg>.<module>`.

```
langsimp/
    prompts.py             DISTILL_SYSTEM_PROMPT (long, for the teacher) and SFT_SYSTEM_PROMPT (baked into training)
    verifier.py            CEFR judge (LocalJudge / OpenRouter), DifficultyRankingTest, PacingVarietyTest, length_ratio_score
    data/
        sources.py         Random Wikipedia paragraph fetcher
        distill.py         Teacher class wrapping OpenRouter; CLI subcommands `sft` and `dpo` (4-strategy diversified)
        mlx_format.py      Carve frozen eval set + convert generated JSONL into mlx-lm format (sft / dpo / grpo)
        audit.py           Audit a JSONL of pairs; reports per-record flags + aggregate stats
    training/
        runner.py          Run mlx-lm / mlx-lm-lora training with W&B + adapter versioning (sft / dpo / grpo)
        rewards.py         GRPO reward components (length / vocab / meaning + repetition + difficulty stubs) plus `audit` and `variety` CLIs
    inference/
        engine.py          Shared model-load + chat-template + clean-generation primitives
        generate.py        Ad-hoc simplification CLI (text arg / --file / stdin)
        eval_harness.py    Run an adapter on the held-out eval set and report % A2 / too-easy / too-hard
        iterate.py         Single-paragraph qualitative prompt-iteration tool

scripts/                   Shell entrypoints for mlx-lm/mlx-lm-lora training (sft / dpo / grpo) + iterate_prompt.py
tests/                     pytest suite mirroring the package (run with `uv run pytest`)
data/                      Generated datasets (gitignored)
adapters/                  Trained LoRA adapters (gitignored)
logs/                      Training logs (gitignored)
eval_results/              Per-adapter eval JSON + summaries (gitignored)
samples.jsonl              CEFR few-shot examples (tracked) — used by the judge prompts
```

## Setup

```bash
uv sync
echo "OPENROUTER_API_KEY=sk-..." > .env   # used by distill.py and (optionally) the GRPO meaning judge
```

## Pipeline

```bash
# 1. Generate SFT pairs (Opus simplifies random Wikipedia paragraphs)
uv run python -m langsimp.data.distill sft --n 200

# 2. Generate DPO pairs (weaker teacher produces 'rejected' completions)
uv run python -m langsimp.data.distill dpo --teacher google/gemma-3-4b-it

# 3. (one time) Carve a frozen held-out evaluation set. The eval prompts
#    are then automatically excluded from every train/valid split below.
uv run python -m langsimp.data.mlx_format carve-eval --n 30

# 4. Convert to mlx-lm format
uv run python -m langsimp.data.mlx_format sft
uv run python -m langsimp.data.mlx_format dpo
uv run python -m langsimp.data.mlx_format grpo    # 134 train + 30 GRPO-valid (separate from eval)

# 5. Train (from repo root). All three go through `langsimp.training.runner`, which forwards
#    metrics to Weights & Biases live. Set WANDB_MODE=disabled to opt out.
bash scripts/train_mlx.sh         # SFT
bash scripts/train_dpo_mlx.sh     # DPO,  resumes from adapters/sft/latest
bash scripts/train_grpo_mlx.sh    # GRPO, resumes from adapters/dpo/latest by default

# GRPO meaning-reward judge — pick one:
#   * OpenRouter (recommended; defaults to anthropic/claude-haiku-latest):
#       MEANING_JUDGE_BACKEND=openrouter bash scripts/train_grpo_mlx.sh
#   * Local LM Studio at http://127.0.0.1:1234/v1:
#       MEANING_JUDGE_URL=http://127.0.0.1:1234/v1 bash scripts/train_grpo_mlx.sh
#   * No judge (meaning reward returns a constant 0.5):
#       bash scripts/train_grpo_mlx.sh
#
# To start GRPO from the base model (skip the DPO resume — useful while
# SFT/DPO are still weak): RESUME_ADAPTER="" bash scripts/train_grpo_mlx.sh

# 6. Evaluate against the frozen held-out set (requires LM Studio).
uv run python -m langsimp.inference.eval_harness --adapter base                  # baseline
uv run python -m langsimp.inference.eval_harness --adapter adapters/sft/latest   # newest SFT adapter (versioned)
uv run python -m langsimp.inference.eval_harness --adapter adapters/dpo/latest   # newest DPO adapter

# 7. Ad-hoc inspection of a single adapter on arbitrary text.
uv run python -m langsimp.inference.generate --adapter adapters/sft/latest --show-source "Complex paragraph..."
cat input.txt | uv run python -m langsimp.inference.generate --adapter adapters/sft/latest
```

## GRPO reward sanity tools

`langsimp.training.rewards` doubles as an offline audit tool. Use it before/after training to verify the rewards are calibrated and producing variance:

```bash
# Score a JSONL of {complex, simple} or {complex, output} records.
# Surfaces the mean per-component score and the worst-scoring records.
uv run python -m langsimp.training.rewards audit data/sft.jsonl
uv run python -m langsimp.training.rewards audit data/sft.jsonl --with-judge --lm-studio-url http://127.0.0.1:1234/v1

# Sample G rollouts per prompt from an adapter and report the per-group
# reward std. GRPO advantage = (reward - mean) / std within a group; if
# std ≈ 0 across most groups, GRPO can't learn — this catches that BEFORE
# burning training compute.
uv run python -m langsimp.training.rewards variety --adapter adapters/sft/latest \
    --n-prompts 5 --group-size 4 --show-rollouts
```

## Verifier / prompt iteration

`langsimp.verifier` runs CEFR-level classification through a chat-completions judge — by default the local LM Studio at `http://127.0.0.1:1234`, but `LocalJudge(api_key=...)` also works against OpenRouter or any OpenAI-compatible endpoint. Pure-Python signals (`PacingVarietyTest`, `length_ratio_score`) need no judge.

Use `langsimp.inference.iterate` to compare prompt variants on a single paragraph (qualitative):

```bash
uv run python -m langsimp.inference.iterate --runs 4
```

Use `langsimp.data.audit` to score a whole dataset and surface the worst examples:

```bash
uv run python -m langsimp.data.audit data/sft.jsonl                  # length + pacing only
uv run python -m langsimp.data.audit data/sft.jsonl --with-judge     # also CEFR-level (needs LM Studio)
```

For A/B-testing prompt edits against the existing dataset:

```bash
uv run python scripts/iterate_prompt.py --n 10
uv run python -m langsimp.data.audit data/sft_new_prompt.jsonl --with-judge
```

## Tests

```bash
uv run pytest
```

Tests do not hit the network or the LM Studio judge — the OpenAI client and the judge are mocked.

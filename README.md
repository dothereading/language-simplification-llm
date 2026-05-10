# Language Simplification Fine-Tuning

![alt text](simple_lm_banner.jpg)

Fine-tune a small Gemma model to rewrite complex English at **CEFR A2** (Elementary) level. See [`PLAN.md`](PLAN.md) for the staged pipeline (data → SFT → DPO → GRPO → eval) and [`CLAUDE.md`](CLAUDE.md) for the development conventions used in this repo.

## Layout

```
prompts.py        DISTILL_SYSTEM_PROMPT (long, for the teacher) and SFT_SYSTEM_PROMPT (short, baked into training)
sources.py        Random Wikipedia paragraph fetcher
distill.py        Teacher class wrapping OpenRouter; CLI subcommands `sft` and `dpo` (4-strategy diversified)
mlx_data.py       Carve frozen eval set + convert generated JSONL into mlx-lm format (sft / dpo / grpo)
train.py          Run mlx-lm / mlx-lm-lora training with W&B + adapter versioning (sft / dpo / grpo)
rewards.py        GRPO reward components (length / vocab / meaning + repetition + difficulty stubs) with mlx-lm-lora @register_reward_function adapters
inference.py      Shared model-load + chat-template + clean-generation primitives
generate.py       Ad-hoc simplification CLI (text arg / --file / stdin)
eval_harness.py   Run an adapter on the held-out eval set and report % A2 / too-easy / too-hard
verifier.py       CEFR judge + reward tests (DifficultyRanking, PacingVariety, length_ratio)
dataset_audit.py  Audit a JSONL of pairs; reports per-record flags + aggregate stats
iterate.py        Single-paragraph qualitative prompt-iteration tool
scripts/          Shell entrypoints for mlx-lm/mlx-lm-lora training (sft / dpo / grpo) + iterate_prompt.py
tests/            pytest suite (run with `uv run pytest`)
data/             Generated datasets (gitignored)
adapters/         Trained LoRA adapters (gitignored)
logs/             Training logs (gitignored)
eval_results/     Per-adapter eval JSON + summaries (gitignored)
```

## Setup

```bash
uv sync
echo "OPENROUTER_API_KEY=sk-..." > .env   # used by distill.py and (optionally) the GRPO meaning judge
```

## Pipeline

```bash
# 1. Generate SFT pairs (Opus simplifies random Wikipedia paragraphs)
uv run python distill.py sft --n 200

# 2. Generate DPO pairs (weaker teacher produces 'rejected' completions)
uv run python distill.py dpo --teacher google/gemma-3-4b-it

# 3. (one time) Carve a frozen held-out evaluation set. The eval prompts
#    are then automatically excluded from every train/valid split below.
uv run python mlx_data.py carve-eval --n 30

# 4. Convert to mlx-lm format
uv run python mlx_data.py sft
uv run python mlx_data.py dpo
uv run python mlx_data.py grpo    # 134 train + 30 GRPO-valid (separate from eval)

# 5. Train (from repo root). All three go through train.py, which forwards
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
uv run python eval_harness.py --adapter base                  # baseline
uv run python eval_harness.py --adapter adapters/sft/latest   # newest SFT adapter (versioned)
uv run python eval_harness.py --adapter adapters/dpo/latest   # newest DPO adapter

# 7. Ad-hoc inspection of a single adapter on arbitrary text.
uv run python generate.py --adapter adapters/sft/latest --show-source "Complex paragraph..."
cat input.txt | uv run python generate.py --adapter adapters/sft/latest
```

## GRPO reward sanity tools

`rewards.py` doubles as an offline audit tool. Use it before/after training to verify the rewards are calibrated and producing variance:

```bash
# Score a JSONL of {complex, simple} or {complex, output} records.
# Surfaces the mean per-component score and the worst-scoring records.
uv run python rewards.py audit data/sft.jsonl
uv run python rewards.py audit data/sft.jsonl --with-judge --lm-studio-url http://127.0.0.1:1234/v1

# Sample G rollouts per prompt from an adapter and report the per-group
# reward std. GRPO advantage = (reward - mean) / std within a group; if
# std ≈ 0 across most groups, GRPO can't learn — this catches that BEFORE
# burning training compute.
uv run python rewards.py variety --adapter adapters/sft/latest \
    --n-prompts 5 --group-size 4 --show-rollouts
```

## Verifier / prompt iteration

`verifier.py` runs CEFR-level classification through a chat-completions judge — by default the local LM Studio at `http://127.0.0.1:1234`, but `LocalJudge(api_key=...)` also works against OpenRouter or any OpenAI-compatible endpoint. Pure-Python signals (`PacingVarietyTest`, `length_ratio_score`) need no judge.

Use `iterate.py` to compare prompt variants on a single paragraph (qualitative):

```bash
uv run python iterate.py --runs 4
```

Use `dataset_audit.py` to score a whole dataset and surface the worst examples:

```bash
uv run python dataset_audit.py data/sft.jsonl                  # length + pacing only
uv run python dataset_audit.py data/sft.jsonl --with-judge     # also CEFR-level (needs LM Studio)
```

For A/B-testing prompt edits against the existing dataset:

```bash
uv run python scripts/iterate_prompt.py --n 10
uv run python dataset_audit.py data/sft_new_prompt.jsonl --with-judge
```

## Tests

```bash
uv run pytest
```

Tests do not hit the network or the LM Studio judge — the OpenAI client and the judge are mocked.

# Language Simplification Fine-Tuning

Fine-tune a small Gemma model to rewrite complex English at **CEFR A2** (Elementary) level. See [`PLAN.md`](PLAN.md) for the staged pipeline (data → SFT → GRPO → DPO → eval) and [`CLAUDE.md`](CLAUDE.md) for the development conventions used in this repo.

## Layout

```
prompts.py        DISTILL_SYSTEM_PROMPT (long, for the teacher) and SFT_SYSTEM_PROMPT (short, baked into training)
sources.py        Random Wikipedia paragraph fetcher
distill.py        Teacher class wrapping OpenRouter; CLI subcommands `sft` and `dpo`
mlx_data.py       Convert generated JSONL into mlx-lm / mlx-lm-lora training format
verifier.py       CEFR judge + reward tests (DifficultyRanking, PacingVariety, length_ratio)
dataset_audit.py  Audit a JSONL of pairs; reports per-record flags + aggregate stats
iterate.py        Single-paragraph qualitative prompt-iteration tool
scripts/          Shell entrypoints for mlx-lm/mlx-lm-lora training, plus iterate_prompt.py
tests/            pytest suite (run with `uv run pytest`)
data/             Generated datasets (gitignored)
adapters/         Trained LoRA adapters (gitignored)
logs/             Training logs (gitignored)
```

## Setup

```bash
uv sync
echo "OPENROUTER_API_KEY=sk-..." > .env   # only needed for distill.py
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

# 5. Train (from repo root)
bash scripts/train_mlx.sh         # SFT
bash scripts/train_dpo_mlx.sh     # DPO, resumes from SFT adapter
```

## Verifier / prompt iteration

`verifier.py` runs CEFR-level classification through a local LM Studio judge (`http://127.0.0.1:1234`) and exposes pure-Python signals (`PacingVarietyTest`, `length_ratio_score`).

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

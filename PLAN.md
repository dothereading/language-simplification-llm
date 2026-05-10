# Project Plan: Language Simplification LLM

## Goal
Fine-tune a small Gemma model to perform language simplification, transforming complex text into **CEFR A2 level (Elementary English)**. The model must optimize for brevity and simplicity while preserving semantic meaning. The project will compare "Non-Thinking" vs "REASONING" modes.

**Base model (committed):** `mlx-community/gemma-3-1b-it-bf16`. Gemma 4 E2B was the original target but is broken in mlx-lm 0.31.3 (k/v projection mismatch). We are not waiting on that.

## Core Pipeline

### Phase 1: Data Preparation (Synthetic/Distillation)
*   [x] Wikipedia source wired up (`langsimp.data.sources`); ArXiv still TODO.
*   [x] Teacher distillation through OpenRouter (`langsimp.data.distill`, `Teacher` class). Opus produces `chosen`; weaker model (Gemma-3-4B) produces `rejected`.
*   [x] SFT dataset: 194 pairs in `data/sft.jsonl` (Opus, original prompt). Future generations use the iterated prompt.
*   [x] DPO preference dataset: 194 triples in `data/dpo.jsonl`.
*   [x] **Distillation prompt iterated**: dropped low-info detail aggressively, killed redundant cappers, raised A2-floor guidance, encouraged adult register and concept-first ordering. A/B test on a fixed 10-paragraph sample showed length-ratio score 0.957→0.982 and judge A2 hit-rate 6/10→7/10.
*   [x] **Data-quality validator** (`langsimp.data.audit`): flags `length_inflated`, `monotonous`, `too_easy`, `too_hard` per record; reports aggregates.
*   [x] **Held-out eval set carved** — `data/eval.jsonl` (30 records). `langsimp.data.mlx_format carve-eval` is deterministic (hash-based) and refuses to overwrite without `--force`. Splits use hash-based assignment (`assign_split`) so the same prompt always lands in the same bucket — SFT-valid and DPO-valid are guaranteed to contain the same prompts (verified: 16/16 overlap, 0 overlap with eval).
*   [ ] Grow SFT dataset (target ≥1k pairs) to combat the val-loss climb seen in the current run. New rows generated after this point use the iterated prompt; eval set stays frozen.

### Phase 2: Supervised Fine-Tuning (SFT)
*   **Engine:** mlx-lm LoRA (`scripts/train_mlx.sh`).
*   **Status:** trained 300 iters; train loss 0.2–0.5, val loss climbing 1.9 → 2.4 → likely overfitting on 90 train rows. Re-run after dataset grows.
*   **Note:** `langsimp.data.mlx_format sft` now defaults to using *all* rows; the existing adapter was trained on a 100-row subset. To reproduce: pass `--n 100`.

### Phase 3: Reinforcement Learning (GRPO)
*   [x] **GRPO data prep** (`langsimp.data.mlx_format grpo`): emits `data/grpo/{train,valid}.jsonl` in mlx-lm-lora's `{prompt, answer, system}` shape; excludes eval prompts; holds out 30 records as a *GRPO valid set* (separate from `data/eval.jsonl`); curriculum-sorts train ascending by source length.
*   [x] **Reward components** (`langsimp.training.rewards`): all four planned, plus a stub for repetition. v1 active = `LengthVsSourceReward` (output/source ratio in [0.8, 1.3]) + `VocabSimplicityReward` (top-3000 wordfreq, allowed=1 uncommon/sentence, severity=0.5; proper-noun-aware; calibrated against chosen/rejected/bad distributions in `data/{sft,dpo}.jsonl` — see `TestVocabSimplicityCalibration`) + `SemanticPreservationReward` (judge call: facts_preserved + no_hallucinations on 1–5 scale). `RepetitionReward` and `SmoothDifficultyReward` are stubbed; activate later. `CombinedReward` does weighted sum (0.5 meaning / 0.25 length / 0.25 vocab) with a *meaning gate*: if meaning < 0.5 the whole reward zeros out. **Note:** the gate is only applied in offline `audit` / `compute_variety`; mlx-lm-lora does its own un-gated weighted sum at training time.
*   [x] **Meaning judge backends** (`verifier.LocalJudge` + `rewards._get_judge`): `MEANING_JUDGE_BACKEND=openrouter` (recommended; default model `anthropic/claude-haiku-latest`, needs `OPENROUTER_API_KEY`) or local LM Studio via `MEANING_JUDGE_URL`. Falls back to a constant 0.5 (no signal, no crash) if neither is set.
*   [x] **GRPO training** (`langsimp.training.runner grpo` + `scripts/train_grpo_mlx.sh`): wraps `mlx_lm_lora.train --train-mode grpo`, with G=2, temp=0.8, lr=1e-6, max_completion=512. Resumes from `adapters/dpo/latest` by default; pass `RESUME_ADAPTER=""` to start GRPO fresh from base (avoids the bf16-overflow loop when policy and reference start mismatched). W&B parser captures train/loss, train/reward_mean, train/reward_std, train/group_reward_std, train/kl, plus per-component reward μ/σ. Adapter versioned to `adapters/grpo/<timestamp>-<sha>/` with meta.json.
*   [x] **Reward sanity tools**: `python -m langsimp.training.rewards audit <jsonl>` scores any dataset with all rewards; `python -m langsimp.training.rewards variety --adapter <path>` samples G rollouts per prompt and reports per-group reward std (the GRPO advantage signal); `compute_variety()` is also exposed as a library helper.
*   [x] 20-iter smoke train completed with `RESUME_ADAPTER=""` and the OpenRouter judge: clean math (loss / KL finite, total reward improved 0.66 → 0.73), but every rollout hit the 512-token completion limit — model needs better SFT before a real run is worthwhile.
*   [ ] Run real GRPO training and eval against the held-out set (gated on stronger SFT/DPO).

### Phase 4: Preference Alignment (DPO)
*   **Engine:** mlx-lm-lora (`scripts/train_dpo_mlx.sh`), resumes from the SFT adapter.
*   **Status:** trained 300 iters with β=0.1; train loss saturated at 0.000 / accuracy 1.0 very early. Suspicious — likely the model is learning surface differences (length, punctuation) between Opus and Gemma-3-4B rather than simplification quality. Needs held-out CEFR-judge evaluation before re-running.

### Phase 5: Held-out Evaluation
We are *not* using SARI, BLEU, or semantic-similarity metrics. Evaluation re-uses the same CEFR-judge approach as `verifier.DifficultyRankingTest`:

*   [x] Frozen evaluation set of 30 paragraphs at `data/eval.jsonl` (deterministic carve, refuses to overwrite).
*   [x] **`langsimp.inference.eval_harness`**: loads base + optional LoRA, runs each held-out prompt, classifies output via `DifficultyRankingTest.classify()` (refactored to expose the level label, not just the binary score). Reports % A2, % too-easy (A1 + <A1), % too-hard (B1 + B2+), mean length ratio, and one sample per category. Persists JSON to `eval_results/<adapter>_<timestamp>.json`.
*   [ ] Run baseline (base / SFT / DPO) once to establish current numbers.

### Observability
*   [x] **W&B wired in** via `langsimp.training.runner`. Wraps the underlying mlx-lm / mlx-lm-lora subprocess, parses log lines with regex, forwards metrics live to W&B (projects `lang-simp-sft` and `lang-simp-dpo`). Run names include timestamp, stage, short model name, iters, lr, beta. Offline-safe: missing API key or `WANDB_MODE=disabled` falls back to plain stdout-only training. Shell scripts now call `langsimp.training.runner`; the env-var override pattern is preserved.

### Adapter management
*   [x] **Versioned outputs.** `langsimp.training.runner` writes adapters to `adapters/<stage>/<timestamp>-<sha>/` by default and updates `adapters/<stage>/latest` symlink on success. `meta.json` next to the weights records: stage, timestamp, git SHA, dataset hash, hyperparameters, full training command, W&B run ID + URL, final train and valid metrics. Pin a fixed dir via `ADAPTER_DIR=...` env or `--adapter-path` (skips versioning). DPO resume default updated to `adapters/sft/latest/...`.

### Inference
*   [x] **`langsimp.inference.generate`** runs any adapter (or `base`) on arbitrary text — positional, `--file`, or stdin. Multi-paragraph input is split on blank lines. Shared model-load / chat-template / output-cleaning primitives live in `langsimp.inference.engine` and are reused by `langsimp.inference.eval_harness`.

## Tech Stack
*   **Base Model:** `mlx-community/gemma-3-1b-it-bf16`
*   **Training Engine:** mlx-lm + mlx-lm-lora (LoRA)
*   **RL Framework:** GRPO (backlog)
*   **Preference Alignment:** DPO via mlx-lm-lora
*   **Reward Sandbox:** OpenAI-compatible chat judge — OpenRouter (default `anthropic/claude-haiku-latest`) or local LM Studio
*   **Observability:** Weights & Biases
*   **Dependency Management:** `uv`
*   **Tests:** `pytest` (mocked OpenAI client + mocked judge)

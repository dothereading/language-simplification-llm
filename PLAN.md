# Project Plan: Language Simplification LLM

## Goal
Fine-tune a small Gemma model to perform language simplification, transforming complex text into **CEFR A2 level (Elementary English)**. The model must optimize for brevity and simplicity while preserving semantic meaning. The project will compare "Non-Thinking" vs "REASONING" modes.

**Base model (committed):** `mlx-community/gemma-3-1b-it-bf16`. Gemma 4 E2B was the original target but is broken in mlx-lm 0.31.3 (k/v projection mismatch). We are not waiting on that.

## Core Pipeline

### Phase 1: Data Preparation (Synthetic/Distillation)
*   [x] Wikipedia source wired up (`sources.py`); ArXiv still TODO.
*   [x] Teacher distillation through OpenRouter (`distill.py`, `Teacher` class). Opus produces `chosen`; weaker model (Gemma-3-4B) produces `rejected`.
*   [x] SFT dataset: 194 pairs in `data/sft.jsonl` (Opus, original prompt). Future generations use the iterated prompt.
*   [x] DPO preference dataset: 194 triples in `data/dpo.jsonl`.
*   [x] **Distillation prompt iterated**: dropped low-info detail aggressively, killed redundant cappers, raised A2-floor guidance, encouraged adult register and concept-first ordering. A/B test on a fixed 10-paragraph sample showed length-ratio score 0.957→0.982 and judge A2 hit-rate 6/10→7/10.
*   [x] **Data-quality validator** (`dataset_audit.py`): flags `length_inflated`, `monotonous`, `too_easy`, `too_hard` per record; reports aggregates.
*   [x] **Held-out eval set carved** — `data/eval.jsonl` (30 records). `mlx_data.py carve-eval` is deterministic (hash-based) and refuses to overwrite without `--force`. Splits use hash-based assignment (`assign_split`) so the same prompt always lands in the same bucket — SFT-valid and DPO-valid are guaranteed to contain the same prompts (verified: 16/16 overlap, 0 overlap with eval).
*   [ ] Grow SFT dataset (target ≥1k pairs) to combat the val-loss climb seen in the current run. New rows generated after this point use the iterated prompt; eval set stays frozen.

### Phase 2: Supervised Fine-Tuning (SFT)
*   **Engine:** mlx-lm LoRA (`scripts/train_mlx.sh`).
*   **Status:** trained 300 iters; train loss 0.2–0.5, val loss climbing 1.9 → 2.4 → likely overfitting on 90 train rows. Re-run after dataset grows.
*   **Note:** `mlx_data.py sft` now defaults to using *all* rows; the existing adapter was trained on a 100-row subset. To reproduce: pass `--n 100`.

### Phase 3: Reinforcement Learning (GRPO) — *backlog*
GRPO loop and reward components are not yet implemented; tracked in the task list. The CEFR-difficulty judge in `verifier.py` will be reward D when this lands.

1.  **Reward A: Length Constraint** — penalty for sentences > 10 words, monotonically increasing.
2.  **Reward B: Vocabulary Simplicity** — penalty when more than 1–2 uncommon words appear in a sentence (frequency-list based).
3.  **Reward C: Semantic Preservation** — independent judge compares source vs. simplification for info loss / hallucination.
4.  **Reward D: Difficulty Ranking** — LM judge labels output as A1 / A2 / B1; reward rewards A2. Implemented in `verifier.DifficultyRankingTest`.

### Phase 4: Preference Alignment (DPO)
*   **Engine:** mlx-lm-lora (`scripts/train_dpo_mlx.sh`), resumes from the SFT adapter.
*   **Status:** trained 300 iters with β=0.1; train loss saturated at 0.000 / accuracy 1.0 very early. Suspicious — likely the model is learning surface differences (length, punctuation) between Opus and Gemma-3-4B rather than simplification quality. Needs held-out CEFR-judge evaluation before re-running.

### Phase 5: Held-out Evaluation
We are *not* using SARI, BLEU, or semantic-similarity metrics. Evaluation re-uses the same CEFR-judge approach as `verifier.DifficultyRankingTest`:

*   Frozen evaluation set of complex paragraphs (carved from `data/sft.jsonl` before any further data generation).
*   For each adapter under test (base, SFT, DPO, GRPO): generate a simplification per held-out prompt, then run `DifficultyRankingTest` to label A1/A2/B1+.
*   Primary metric: **% of outputs labeled A2** by the judge.
*   Secondary signals: length distribution, qualitative samples.

### Observability
*   Wire **Weights & Biases** into both training scripts so loss / val-loss / DPO accuracy / DPO margin stream live, with one project per training mode (e.g. `lang-simp-sft`, `lang-simp-dpo`). API key in `.env` as `WANDB_API_KEY`.

### Adapter management
*   Adapters currently overwrite `adapters/sft-a2/` and `adapters/dpo-a2/`. Move to per-run directories named with timestamp + key hyperparams + git SHA, with a `meta.json` next to the weights recording the dataset hash, hyperparameters, and run ID. Pin a `latest` symlink for convenience.

### Inference
*   Add `generate.py --adapter <path> "complex paragraph…"` so we can run any adapter on arbitrary text. Needed for debugging and ad-hoc qualitative review. *Deferred to after eval harness lands.*

## Tech Stack
*   **Base Model:** `mlx-community/gemma-3-1b-it-bf16`
*   **Training Engine:** mlx-lm + mlx-lm-lora (LoRA)
*   **RL Framework:** GRPO (backlog)
*   **Preference Alignment:** DPO via mlx-lm-lora
*   **Reward Sandbox:** local LM Studio judge (CEFR difficulty ranking)
*   **Observability:** Weights & Biases
*   **Dependency Management:** `uv`
*   **Tests:** `pytest` (mocked OpenAI client + mocked judge)

#!/usr/bin/env bash
# GRPO fine-tune via mlx-lm-lora, with W&B logging and metric forwarding.
# Resumes from the DPO LoRA adapter at adapters/dpo/latest by default.
# Reward functions live in rewards.py (length_reward, vocab_reward,
# meaning_reward). The meaning reward picks a judge from env:
#   * MEANING_JUDGE_BACKEND=openrouter + OPENROUTER_API_KEY
#       → hosted judge (default model: anthropic/claude-haiku-latest)
#   * MEANING_JUDGE_URL set (no backend flag)
#       → local LM Studio / vLLM at that URL
#   * neither set
#       → meaning_reward returns a constant 0.5 (no signal, no crash)
#
# Run from the repo root:
#   uv run python -m langsimp.data.mlx_format grpo
#   bash scripts/train_grpo_mlx.sh
#
# Env-var overrides (any subset):
#   MODEL DATA_DIR ADAPTER_DIR RESUME_ADAPTER ITERS BATCH LR LORA_LAYERS
#   GROUP_SIZE TEMPERATURE MAX_COMPLETION_LENGTH WANDB_PROJECT
#   MEANING_JUDGE_BACKEND MEANING_JUDGE_URL MEANING_JUDGE_MODEL
#   OPENROUTER_API_KEY
#
# To run without W&B: WANDB_MODE=disabled bash scripts/train_grpo_mlx.sh
set -euo pipefail

cd "$(dirname "$0")/.."

ARGS=(grpo
    --model "${MODEL:-mlx-community/gemma-3-1b-it-bf16}"
    --data "${DATA_DIR:-data/grpo}"
    --resume-adapter "${RESUME_ADAPTER-adapters/dpo/latest/adapters.safetensors}"
    --iters "${ITERS:-200}"
    --batch-size "${BATCH:-1}"
    --lr "${LR:-1e-6}"
    --lora-layers "${LORA_LAYERS:-16}"
    --group-size "${GROUP_SIZE:-2}"
    --temperature "${TEMPERATURE:-0.8}"
    --max-completion-length "${MAX_COMPLETION_LENGTH:-512}"
    --project "${WANDB_PROJECT:-lang-simp-grpo}")

# Optional: pin a specific adapter dir (skips versioning + latest symlink).
if [[ -n "${ADAPTER_DIR:-}" ]]; then
    ARGS+=(--adapter-path "$ADAPTER_DIR")
fi

uv run python -m langsimp.training.runner "${ARGS[@]}"

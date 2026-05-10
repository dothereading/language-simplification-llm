#!/usr/bin/env bash
# DPO fine-tune via mlx-lm-lora, with W&B logging and metric forwarding.
# `chosen`  = Opus simplification (from data/sft.jsonl)
# `rejected` = Gemma-3-4B simplification (from data/dpo.jsonl)
#
# Routes through langsimp.training.runner so loss / accuracy / reward margin land in
# Weights & Biases live.
#
# Run from the repo root:
#   uv run python -m langsimp.data.mlx_format dpo
#   bash scripts/train_dpo_mlx.sh
#
# Env-var overrides (any subset):
#   MODEL DATA_DIR ADAPTER_DIR RESUME_ADAPTER ITERS BATCH LR LORA_LAYERS BETA WANDB_PROJECT
#
# To run without W&B: WANDB_MODE=disabled bash scripts/train_dpo_mlx.sh
set -euo pipefail

cd "$(dirname "$0")/.."

ARGS=(dpo
    --model "${MODEL:-mlx-community/gemma-3-1b-it-bf16}"
    --data "${DATA_DIR:-data/dpo_mlx}"
    --resume-adapter "${RESUME_ADAPTER:-adapters/sft/latest/adapters.safetensors}"
    --iters "${ITERS:-300}"
    --batch-size "${BATCH:-1}"
    --lr "${LR:-5e-6}"
    --lora-layers "${LORA_LAYERS:-16}"
    --beta "${BETA:-0.1}"
    --project "${WANDB_PROJECT:-lang-simp-dpo}")

# Optional: pin a specific adapter dir (skips versioning + latest symlink).
if [[ -n "${ADAPTER_DIR:-}" ]]; then
    ARGS+=(--adapter-path "$ADAPTER_DIR")
fi

uv run python -m langsimp.training.runner "${ARGS[@]}"

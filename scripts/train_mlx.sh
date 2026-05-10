#!/usr/bin/env bash
# Fine-tune the SFT LoRA via mlx-lm, with W&B logging and metric forwarding.
# Routes through train.py (Python wrapper) so we get a parsed loss curve in
# Weights & Biases as well as the raw stdout in the terminal.
#
# Run from the repo root:
#   bash scripts/train_mlx.sh
#
# Env-var overrides (any subset):
#   MODEL DATA_DIR ADAPTER_DIR ITERS BATCH LR LORA_LAYERS WANDB_PROJECT
#
# To run without W&B: WANDB_MODE=disabled bash scripts/train_mlx.sh
set -euo pipefail

cd "$(dirname "$0")/.."

ARGS=(sft
    --model "${MODEL:-mlx-community/gemma-3-1b-it-bf16}"
    --data "${DATA_DIR:-data/mlx}"
    --iters "${ITERS:-300}"
    --batch-size "${BATCH:-1}"
    --lr "${LR:-1e-4}"
    --lora-layers "${LORA_LAYERS:-16}"
    --project "${WANDB_PROJECT:-lang-simp-sft}")

# Optional: pin a specific adapter dir (skips versioning + latest symlink).
if [[ -n "${ADAPTER_DIR:-}" ]]; then
    ARGS+=(--adapter-path "$ADAPTER_DIR")
fi

uv run python -m langsimp.training.runner "${ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${1:-/opt/gemma4_learning_agent}"
MODEL="${PROJECT_DIR}/models/gemma/gemma-4-12B-it"
ADAPTER="${PROJECT_DIR}/trained_models/gemma4_12b_lora/adapter"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate gemma4_vllm

ARGS=(
  "${MODEL}"
  --host 0.0.0.0
  --port 8001
  --dtype bfloat16
  --gpu-memory-utilization 0.88
  --max-model-len 8192
)

if [ -d "${ADAPTER}" ]; then
  echo "Loading LoRA Adapter: ${ADAPTER}"
  ARGS+=(--enable-lora --lora-modules "gemma4-learning=${ADAPTER}")
else
  echo "Adapter not found. Starting base Gemma4-12B."
fi

vllm serve "${ARGS[@]}"

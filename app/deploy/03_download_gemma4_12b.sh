#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${1:-/opt/gemma4_learning_agent}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate gemma4_vllm
mkdir -p "${PROJECT_DIR}/models/gemma"
echo "请先完成 Hugging Face Gemma 访问授权，然后按提示登录。"
hf auth login
hf download google/gemma-4-12b-it --local-dir "${PROJECT_DIR}/models/gemma/gemma-4-12B-it"

#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n gemma4_vllm python=3.10 -y || true
conda activate gemma4_vllm
pip install -U pip
pip install vllm
echo "vLLM environment ready: gemma4_vllm"

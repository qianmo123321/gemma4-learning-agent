#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${1:-/opt/gemma4_learning_agent}"
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n gemma4_web python=3.10 -y || true
conda activate gemma4_web
pip install -U pip
pip install -r "${PROJECT_DIR}/backend/requirements.txt"
echo "Web environment ready: gemma4_web"

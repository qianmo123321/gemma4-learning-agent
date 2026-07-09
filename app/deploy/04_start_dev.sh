#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${1:-/opt/gemma4_learning_agent}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate gemma4_web
cd "${PROJECT_DIR}/backend"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

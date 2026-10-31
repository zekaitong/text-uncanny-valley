#!/bin/bash
#SBATCH --job-name=gemini-legal-800
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-7%4
#SBATCH --output=rebuttal/gemini_legal_11point/logs/%A_%a.out
#SBATCH --error=rebuttal/gemini_legal_11point/logs/%A_%a.err
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
cd "${SLURM_SUBMIT_DIR:-$PWD}/rebuttal/gemini_legal_11point"
mkdir -p logs
exec "${PYTHON_BIN:-python3}" worker.py --run --index "$SLURM_ARRAY_TASK_ID" --plan "${PLAN_ID:-initial}"

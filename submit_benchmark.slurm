#!/bin/bash
#SBATCH --job-name=absence_bench
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

export OPENAI_API_BASE="https://api.openai.com/v1"
export OPENAI_BASE_URL="https://api.openai.com/v1"

python run_benchmark.py --config $CONFIG_FILE
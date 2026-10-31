#!/bin/bash
#SBATCH --job-name=math_uncanny
#SBATCH --output=logs/%j_out.log
#SBATCH --error=logs/%j_err.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00

python -u gsm8k_test/math_test.py \
    --models gemini-3-flash-preview gpt-5.2 claude-haiku-4-5 \
    --sample_size 500 \
    --batch_size 10 \
    --rates 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --seed 42
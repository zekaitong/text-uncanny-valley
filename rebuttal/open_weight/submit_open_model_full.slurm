#!/bin/bash
#SBATCH --job-name=open_model_full
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=rebuttal/open_weight/logs/%x_%j.out
#SBATCH --error=rebuttal/open_weight/logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p rebuttal/open_weight/logs
export TMPDIR="${TMPDIR:-/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "${EXTRA_PYTHONPATH:-}" ]; then
  export PYTHONPATH="${EXTRA_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
fi

EXTRA_ARGS=()
if [ -n "${MODEL_PATH:-}" ]; then
  EXTRA_ARGS+=(--model-path "${MODEL_PATH}")
fi
if [ "${ALLOW_DOWNLOAD:-0}" = "1" ]; then
  EXTRA_ARGS+=(--allow-download)
fi
if [ -n "${RATES:-}" ]; then
  EXTRA_ARGS+=(--rates "${RATES}")
fi
if [ -n "${MATH_RATES:-}" ]; then
  EXTRA_ARGS+=(--math-rates "${MATH_RATES}")
fi
if [ -n "${NUM_SHARDS:-}" ]; then
  EXTRA_ARGS+=(--num-shards "${NUM_SHARDS}")
fi
if [ -z "${SHARD_INDEX:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"
fi
if [ -n "${SHARD_INDEX:-}" ]; then
  EXTRA_ARGS+=(--shard-index "${SHARD_INDEX}")
fi

"${PYTHON_BIN:-python3}" \
  rebuttal/open_weight/open_model_full_runner.py \
  --model-key "${MODEL_KEY}" \
  --absence-sample-size "${ABSENCE_SAMPLE_SIZE:-100}" \
  --math-sample-size "${MATH_SAMPLE_SIZE:-500}" \
  --seed "${SEED:-42}" \
  --output-base "${OUTPUT_BASE:-outputs/open_model_full}" \
  --tasks "${TASKS:-absencebench,niah,math_gsm8k}" \
  --absence-domains "${ABSENCE_DOMAINS:-ld,gp,gsm}" \
  --niah-domains "${NIAH_DOMAINS:-ld,gp,gsm}" \
  --absence-max-new-tokens "${ABSENCE_MAX_NEW_TOKENS:-512}" \
  --math-max-new-tokens "${MATH_MAX_NEW_TOKENS:-1024}" \
  "${EXTRA_ARGS[@]}"

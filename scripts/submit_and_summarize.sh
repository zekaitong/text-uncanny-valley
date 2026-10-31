#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config_yaml> <sweep_root_dir> [partition]"
  echo "Example: $0 configs/ld_seed42.yaml data/legal_docs_random_slot_seed42 general"
  exit 1
fi

CONFIG_FILE="$1"
SWEEP_ROOT="$2"
PARTITION="${3:-general}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config file not found: $CONFIG_FILE"
  exit 1
fi

mkdir -p "logs"
mkdir -p "$SWEEP_ROOT"

echo "Submitting job with --wait..."
JOB_ID=$(sbatch --parsable --wait --partition="$PARTITION" --export=ALL,CONFIG_FILE="$CONFIG_FILE" submit_benchmark.slurm)
echo "Job finished: $JOB_ID"

echo "Summarizing sweep results under: $SWEEP_ROOT"
python summaries/summarize_f1.py "$SWEEP_ROOT" \
  --format table \
  --csv_out "$SWEEP_ROOT/summary.csv" \
  --json_out "$SWEEP_ROOT/summary.json"

echo "Summary written:"
echo "  - $SWEEP_ROOT/summary.csv"
echo "  - $SWEEP_ROOT/summary.json"

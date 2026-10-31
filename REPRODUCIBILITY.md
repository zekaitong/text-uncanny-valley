# Reproducibility Guide (NeurIPS Submission)

This document maps the paper protocol to runnable code in this repository.

## Core Evaluation Setup

- Task: Absence detection using original vs modified document contexts
- Omission probability: `--prob 0.1`
- Metric: Micro-F1 over omitted lines
- Sample size: `--sample_size 100`
- Random seed for sampling/omission: `--random_seed 42`
- Fragmentation sweep: `0.0, 0.1, ..., 1.0`
- Variance reporting: mean and standard deviation across 3 fragmentation seeds

## Pipeline Entry Points

- End-to-end runner: `run_benchmark.py`
- Omission processing only: `dataset_construction/process_data.py`
- LLM evaluation only: `tests/test_llms.py`
- Plotting/aggregation: scripts in `plot/`, plus `summaries/summarize_f1.py`, `summaries/summarize_entropy.py`

## Required Environment Variables

Set only the keys you need:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export TOGETHER_API_KEY=...
```

## Minimal End-to-End Command

```bash
python run_benchmark.py \
  --input_file data/legal_docs.jsonl \
  --document_type legal_documents \
  --prob 0.1 \
  --sample_size 100 \
  --random_seed 42 \
  --provider_models google:gemini-3.0-flash \
  --batch_size 5 \
  --output_dir legal_docs_seed42
```

## Running Multiple Models

```bash
python run_benchmark.py \
  --input_file data/legal_docs.jsonl \
  --document_type legal_documents \
  --prob 0.1 \
  --sample_size 100 \
  --random_seed 42 \
  --provider_models google:gemini-3.0-flash openai:gpt-5.2 anthropic:claude-4.5-haiku \
  --batch_size 5 \
  --output_dir legal_docs_main
```

## NIAH Added-line Control

Use `--use_needle` plus a needle file from `data/needle/` to switch from omission detection to added-line detection:

```bash
python run_benchmark.py \
  --input_file data/legal_docs.jsonl \
  --document_type legal_documents \
  --use_needle \
  --needles_file data/needle/ld/legal_needles.txt \
  --prob 0.1 \
  --sample_size 100 \
  --random_seed 42 \
  --provider_models google:gemini-3.0-flash \
  --batch_size 5 \
  --output_dir legal_docs_niah_seed42/ori
```

For WFR-aligned NIAH sweeps, choose matching `*_waX.txt` needle files per condition (e.g., `legal_needles_wa1.txt` ... `legal_needles_wa10.txt`).

To recompute corrected NIAH metrics from existing artifacts (no new API calls):

```bash
python summaries/recompute_niah_scores.py results/NIAH --output_suffix .fixed2
python summaries/build_niah_summary_from_results.py results/NIAH --result_file_name results.fixed2.json --output_dir summaries_fixed2
```

## Experiment C (Standalone GSM8K Reasoning)

Run via `gsm8k_test/math_test.py` (independent from AbsenceBench omission pipeline).
For paper-facing evaluation, use strict answer parsing:

```bash
python gsm8k_test/math_test.py \
  --models gemini-3-flash-preview gpt-5.2 claude-haiku-4-5 \
  --sample_size 500 \
  --batch_size 5 \
  --seed 42 \
  --parse_mode strict
```

Note: `--parse_mode strict` and `--parse_mode legacy` are different scoring protocols.
Do not compare numbers across modes without explicitly stating the parsing mode.

## Suggested Three-Seed Protocol

Run the same command with three `--random_seed` values for fragmentation/evaluation consistency (for example: `42`, `43`, `44`) and aggregate with existing summary scripts.

## Output Artifacts

For each run directory in `data/<output_dir>/`:

- `processed_data.jsonl`: processed documents with omissions
- `used_data.jsonl`: sampled subset actually evaluated
- `results.json`: per-model detailed predictions and aggregate Micro-F1

## Notes

- API model availability can change over time; pin model names in config files for reproducibility.
- Costs and latency depend on model/provider and prompt length.
- Keep raw run outputs for auditability during rebuttal/camera-ready preparation.


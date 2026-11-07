# Reproducibility Guide (NeurIPS 2026 Evaluations & Datasets Track)

This document maps the paper protocol to runnable code in this repository.

## Core Evaluation Setup

- Task: Absence detection using original vs modified document contexts
- Target word-fragmentation intensity: `spacing_transform.py --intensity <rate>`
- Fragmentation seed: `spacing_transform.py --seed <seed>`
- Omission probability: `run_benchmark.py --prob 0.1`
- Metric: Micro-F1 over omitted lines
- Sample size: `--sample_size 100`
- Random seed for sampling and omission/needle selection: `run_benchmark.py --random_seed 42`
- Target fragmentation sweep: `0.0, 0.1, ..., 1.0`
- Variance reporting: mean and standard deviation across 3 fragmentation seeds

`--prob` is not a fragmentation parameter. Nonzero-WFR inputs must first be generated with `transformations/spacing_transform.py`. The transformation records the realized per-document `word_fragmentation_rate` in JSONL metadata; stochastic realization can differ slightly from the target `--intensity`.

## Pipeline Entry Points

- Word-fragmentation input generation: `transformations/spacing_transform.py`
- Omission construction plus model evaluation: `run_benchmark.py`
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

## Minimal End-to-End Condition

First generate a target WFR = 0.3 input:

```bash
python transformations/spacing_transform.py \
  --input data/legal_docs.jsonl \
  --mode intra_word \
  --intensity 0.3 \
  --seed 42 \
  --suffix _wfr_0_3 \
  --output_dir data/transformed/seed42
```

Then apply line omission and run evaluation:

```bash
python run_benchmark.py \
  --input_file data/transformed/seed42/legal_docs_wfr_0_3.jsonl \
  --document_type legal_documents \
  --prob 0.1 \
  --sample_size 100 \
  --random_seed 42 \
  --provider_models google:gemini-3.0-flash \
  --batch_size 5 \
  --output_dir legal_docs_seed42/p_0_3
```

## Running Multiple Models

```bash
python run_benchmark.py \
  --input_file data/transformed/seed42/legal_docs_wfr_0_3.jsonl \
  --document_type legal_documents \
  --prob 0.1 \
  --sample_size 100 \
  --random_seed 42 \
  --provider_models google:gemini-3.0-flash openai:gpt-5.2 anthropic:claude-4.5-haiku \
  --batch_size 5 \
  --output_dir legal_docs_seed42/p_0_3
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

For WFR-aligned NIAH sweeps, both the source document JSONL and the needle pool must correspond to the same WFR condition. Choose the matching `*_waX.txt` needle file for each transformed input (for example, `legal_needles_wa1.txt` through `legal_needles_wa10.txt`).

### NIAH Scoring Implementations

The repository contains three intentionally distinct NIAH scoring paths:

1. The current API evaluator in `tests/test_llms.py` uses strict line-level matching after lowercasing and trimming outer whitespace. It uses `Counter` multiplicities, and every nonempty response line not consumed by a needle match is one false positive.
2. `summaries/recompute_niah_scores.py` recalculates strict line-level metrics from saved responses without calling a model API. It is a separate implementation and should not be assumed byte-for-byte equivalent to online preprocessing. By default it writes a new suffixed result file rather than overwriting `results.json`.
3. Open-weight runs use the frozen historical evaluator under `rebuttal/open_weight/legacy_evaluator/`. Its historical matching behavior is preserved for provenance and is not interchangeable with the current strict scorer.

An earlier top-level API evaluator could count a returned repeated original-context line once in shared AbsenceBench duplicate-line preprocessing and again in strict NIAH accounting. The current evaluator disables that AbsenceBench prepass in NIAH mode, so each unmatched output line is counted once by the strict NIAH branch. This repository does not include historical result artifacts, so it does not independently establish the numerical impact of that edge case on previously reported runs.

To recalculate NIAH metrics from existing artifacts without new API calls:

```bash
python summaries/recompute_niah_scores.py results/NIAH --output_suffix .fixed2
python summaries/build_niah_summary_from_results.py results/NIAH --result_file_name results.fixed2.json --output_dir summaries_fixed2
```

The documented summary command consumes `results.fixed2.json`; it does not silently fall back to raw `results.json`.

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

Generate each target WFR with three `spacing_transform.py --seed` values and aggregate the resulting condition summaries. The downstream `run_benchmark.py --random_seed` controls sampling and omission/needle selection, not fragmentation. Record both seeds for every run rather than treating them as one parameter.

## Output Artifacts

For each run directory in `data/<output_dir>/`:

- `processed_data.jsonl`: processed documents with omissions
- `used_data.jsonl`: sampled subset actually evaluated
- `results.json`: per-model detailed predictions and aggregate Micro-F1

## Notes

- API model availability can change over time; pin model names in config files for reproducibility.
- Costs and latency depend on model/provider and prompt length.
- Preserve raw model responses and record the scorer implementation used when reporting recalculated metrics.


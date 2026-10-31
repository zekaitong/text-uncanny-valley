# The Text Uncanny Valley (NeurIPS 2026)

Code for the NeurIPS 2026 submission **"The Text Uncanny Valley: Non-Monotonic Performance Degradation in LLM Absence Detection."**

This codebase is **derived from AbsenceBench** and extended for the experiments reported in this paper (main text + appendix).  
This README intentionally documents only the experiments described in the paper.

## Quick Start

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set API keys for the providers you evaluate:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export TOGETHER_API_KEY=...
```

### 2) Input Data Format

Input should be JSONL where each row is one document:

```json
{"id": "doc-1", "lines": ["line A", "line B", "line C"], "metadata": {"source": "demo"}}
```

Required field: `lines` (list of strings). Optional: `id`, `metadata`.

Three-domain inputs typically live under `data/`:

| File | Domain |
|------|--------|
| `data/legal_docs.jsonl` | Legal documents |
| `data/github_prs_extracted.jsonl` | GitHub PR diffs |
| `data/gsm8k_n80.jsonl` | GSM8K long-document construct |
| `data/examples.txt` | Optional few-shot ICL prefix passed via `--icl_file` |
| `data/needle/` | Optional NIAH needle pools (extra lines injected into modified context) |

Licenses and citations for each source are in `DATA.md`.

If you run the ICL experiment, keep the prompt prefix in `data/examples.txt` and pass:

```bash
python run_benchmark.py ... --icl_file data/examples.txt
```

If you run the NIAH control, pass `--use_needle` and a `--needles_file` under `data/needle/`.
Example:

```bash
python run_benchmark.py \
  --input_file data/legal_docs.jsonl \
  --document_type legal_documents \
  --use_needle \
  --needles_file data/needle/ld/legal_needles.txt \
  --prob 0.1 \
  --sample_size 100 \
  --provider_models google:gemini-3.0-flash \
  --output_dir legal_docs_niah/ori
```

`data/needle/` naming convention in this repo:
- `*_needles.txt`: base needle sentences for a domain.
- `*_wa1.txt ... *_wa10.txt`: needle text already fragmented at increasing whitespace anomaly levels, typically aligned with WFR sweep steps.

### 3) Run One Condition End-to-End

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

Outputs are written to `data/<output_dir>/`:
- `processed_data.jsonl`: omission-processed inputs
- `used_data.jsonl`: sampled documents used in evaluation
- `results.json`: model outputs and Micro-F1 metrics

## Experiments Covered in Paper

### Main Text

- Main random-slot fragmentation curves across 3 domains
- Thinking vs baseline (legal documents)
- ICL vs baseline (Gemini, legal documents)
- Fixed-slot fragmentation control
- Tokenization entropy analysis
- Math reasoning under fragmentation (single-problem GSM8K)

### Appendix

- NIAH-style added-line control
- Non-U-shape model results
- Per-condition tables and prompt templates

### Experiment C (Standalone GSM8K Reasoning)

Use the dedicated script (independent from `run_benchmark.py`) with strict final-answer parsing:

```bash
python gsm8k_test/math_test.py \
  --models gemini-3-flash-preview gpt-5.2 claude-haiku-4-5 \
  --sample_size 500 \
  --batch_size 5 \
  --seed 42 \
  --parse_mode strict
```

`--parse_mode strict` accepts only the `#### [number]` final marker for scoring.

## Running Sweeps + Summarization

You currently run each condition with `sbatch` and then summarize; this workflow is now integrated.

- Existing style (unchanged): submit with `submit_benchmark.slurm`
- New integrated style: submit and automatically summarize after completion

```bash
make run_and_summarize \
  CONFIG_FILE=configs/config.yaml \
  SWEEP_ROOT=data/legal_docs_seed42
```

Or directly:

```bash
./scripts/submit_and_summarize.sh configs/config.yaml data/legal_docs_seed42 general
```

This writes:
- `data/<sweep_root>/summary.csv`
- `data/<sweep_root>/summary.json`

`summary.json` is designed for plotting scripts.

### Important: WFR-Aware `output_dir` Naming

`summaries/summarize_f1.py` extracts fragmentation rate from subfolder names.  
When running WFR sweeps, set `output_dir` in each config to a parseable suffix.

Recommended naming:
- `.../ori` (for WFR = 0.0 baseline)
- `.../p_0_1`, `.../p_0_2`, ..., `.../p_1_0`

Example:
```yaml
output_dir: "legal_docs_seed42/p_0_3"
```

If folder names are not parseable, `summaries/summarize_f1.py` will skip them and print warnings.

### NIAH Recompute + Plot (Fixed Scoring)

If you already have NIAH outputs and want corrected FP/Micro-F1 without re-calling model APIs:

```bash
python summaries/recompute_niah_scores.py results/NIAH --output_suffix .fixed2
python summaries/build_niah_summary_from_results.py results/NIAH --result_file_name results.fixed2.json --output_dir summaries_fixed2
python plot/plot_from_summary.py \
  --summary_json results/NIAH/summaries_fixed2/ld_niah_summary.fixed2.json \
  --title "Legal Documents" \
  --output_prefix assets/figures/uncanny_valley_ld_niah
```

### One-command Utilities (Makefile)

```bash
# Submit a SLURM run and summarize sweep outputs
make run_and_summarize CONFIG_FILE=configs/config.yaml SWEEP_ROOT=data/legal_docs_seed42

# Experiment C (standalone GSM8K) with strict answer parsing
make gsm_reasoning_strict

# Recompute corrected NIAH metrics and rebuild summary JSONs
make niah_fix

# Offline checks (no external model API calls)
make offline_test
make offline_test_full
```

## Plotting From Results (No Hardcoded Arrays)

All plot scripts now read structured outputs instead of hardcoded Python lists.

Example: main U-curve (3 seeds -> mean/std)
```bash
python plot/u_curve.py \
  --seed_summaries data/ld_seed42/summary.json data/ld_seed666/summary.json data/ld_seed2026/summary.json \
  --title "Text Uncanny Valley — legal_docs" \
  --output_prefix uncanny_valley_ld
```

Example: entropy overlay
```bash
python summaries/summarize_entropy.py data/ld_seed42 --tokenizer gpt --json_out data/ld_seed42/entropy.json
python plot/u_curve_vs_tokenization_entropy.py \
  --seed_summaries data/ld_seed42/summary.json data/ld_seed666/summary.json data/ld_seed2026/summary.json \
  --entropy_json data/ld_seed42/entropy.json \
  --output_prefix uncanny_valley_ld_entropy
```

## Reproducing Main Paper Protocol

The main experiments in the paper use:
- omission probability `p_omit = 0.1`;
- 100 documents per domain (fixed seed);
- word fragmentation rates `0.0, 0.1, ..., 1.0`;
- three fragmentation seeds (mean ± 1 std across seeds).

Use:
- `transformations/spacing_transform.py` and related scripts for perturbation
- `run_benchmark.py` for omission + evaluation
- `summaries/summarize_f1.py` for sweep aggregation
- `plot/plot_from_summary.py` for figure generation from summarized results

Detailed reproducibility notes are in `REPRODUCIBILITY.md`.

## Slurm Usage

The repository includes `Makefile` + `submit_benchmark.slurm`.

1. Fill `.env`:
```bash
CONDA_ENV=your_env_name
SLURM_PARTITION=your_partition
```

2. Prepare YAML config (see `configs/config.yaml`).

3. Submit:
```bash
make run_benchmark CONFIG_FILE=configs/config.yaml
```

4. Submit and summarize in one step:
```bash
make run_and_summarize CONFIG_FILE=configs/config.yaml SWEEP_ROOT=data/legal_docs_seed42
```

## Repository Layout

- `run_benchmark.py`: end-to-end pipeline (process + evaluate)
- `tests/test_llms.py`: core API evaluation runner + Micro-F1 computation
- `dataset_construction/`: build omission/needle datasets
- `transformations/`: text corruption/fragmentation transforms
- `plot/plot_from_summary.py`: generic plotting from `summary.json`
- `plot/*.py`: plotting utilities that read externally supplied summaries
- `gsm8k_test/math_test.py`: standalone GSM8K fragmentation reasoning evaluation
- `configs/`: run configs for local/cluster execution
- `rebuttal/`: open-weight experiments, Dictionary Repair, Gemini Legal
  11-point completion, and mechanistic probing

## Rebuttal and Appendix Experiments

The additional experiments reported during rebuttal and in the appendices are
released under [`rebuttal/`](rebuttal/README.md). This includes
the eight Gemini Legal rates added to complete the final 11-point sweep in
Appendix I.3. The public artifact contains code, frozen configuration,
prompts, scorers, and dictionaries, but no experiment outputs or result data.
Raw benchmark documents, API responses, cluster logs, model weights, and
credentials are also excluded.

## Release Scope

- Do not include API keys, private endpoints, or internal paths in shared artifacts.
- Keep large outputs (`data/`, `results/`, logs) out of the submission archive unless required.
- Keep only code and small metadata needed to reproduce commands in the paper.
- This repository is not fully self-contained: running full experiments requires commercial LLM APIs and third-party datasets with their own licenses; this artifact documents and scripts those dependencies for reproducibility.

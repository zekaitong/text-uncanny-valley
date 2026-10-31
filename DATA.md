# Data and Assets

This project evaluates absence detection under controlled text fragmentation.

## Third-party raw data

Place compliant snapshots under `data/` (or anywhere else) and set `input_file` / `--input_file` in your YAML or CLI to match.

When you redistribute this project **together with** copies or derivatives of those datasets, you must comply with each license (attribution; for CC BY-SA, share-alike on adaptations—see below). This file summarizes obligations; the authoritative text is always on each dataset’s official page.

### Canonical paths used in this repository

After placing inputs locally, this checkout expects JSONL inputs roughly aligned with the three domains:

| Path | Domain |
|------|--------|
| `data/legal_docs.jsonl` | Legal documents (CUAD-QA–derived pipeline inputs) |
| `data/github_prs_extracted.jsonl` | GitHub pull requests (AbsenceBench `github_prs`–style extraction) |
| `data/gsm8k_n80.jsonl` | GSM8K-derived long documents (e.g. 80-problem concatenation per doc for AbsenceBench) |
| `data/examples.txt` | Optional ICL few-shot prefix text used by `--icl_file` (prompt asset, not dataset) |
| `data/needle/` | Optional NIAH needle pools used by `--use_needle --needles_file ...` |

The default `configs/config.yaml` uses `data/legal_docs.jsonl`. Switch `input_file` and `--document_type` when running PR or GSM8K domains so prompts describe the modality accurately.

Outputs from benchmarking remain under `data/<output_dir>/` (see `README.md`) and are separate from these raw inputs.

For ICL runs, keep examples in plain text (`data/examples.txt`) and pass the path through `run_benchmark.py --icl_file ...` or `icl_file` in YAML. The file is loaded as raw text and prepended to the user prompt.

For NIAH runs, `data/needle/` stores candidate inserted lines ("needles"). `dataset_construction/process_data.py` samples from the selected `--needles_file` and inserts those lines into the modified context; `tests/test_llms.py` then evaluates detection of added lines (not omitted lines).
These needle files are experiment assets in this repository (not a third-party raw dataset source).

In this checkout, needle files are organized by domain and optional whitespace-anomaly level:
- Base files: `legal_needles.txt`, `prs_needles.txt`, `gsm_needles.txt`
- Domain folders: `data/needle/ld/`, `data/needle/gp/`, `data/needle/gsm/`
- `*_wa1.txt` ... `*_wa10.txt`: pre-fragmented needle variants at increasing spacing anomaly levels, typically used to align NIAH needle style with sweep condition.

---

### 1) Legal documents — CUAD / CUAD-QA

- **Role in the paper:** Legal-domain AbsenceBench-style evaluation (long contracts).
- **Typical source:** Hugging Face community mirrors built from CUAD (e.g. [chenghao/cuad_qa](https://huggingface.co/datasets/chenghao/cuad_qa)); upstream CUAD is documented by [The Atticus Project](https://www.atticusprojectai.org/cuad).
- **License (as stated on dataset cards):** **CC BY 4.0** — commercial and non-commercial use allowed; you must provide **attribution**, link to the license, and indicate if you changed the material.
- **Note:** Dataset documentation states that annotators make **no warranty about the license status of underlying EDGAR contracts** (public filings); use remains subject to their disclaimer and privacy policy on [atticusprojectai.org](https://www.atticusprojectai.org/).

**Suggested citation (CUAD paper):**

```bibtex
@article{hendrycks2021cuad,
  title={CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
  author={Dan Hendrycks and Collin Burns and Anya Chen and Spencer Ball},
  journal={arXiv preprint arXiv:2103.06268},
  year={2021}
}
```

---

### 2) Math — GSM8K

- **Role in the paper:** (i) long-document GSM8K-based setup for the main domain panel, and (ii) optional single-problem reasoning evaluation via `gsm8k_test/math_test.py`.
- **Typical source:** [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) on Hugging Face; upstream [openai/grade-school-math](https://github.com/openai/grade-school-math).
- **License (as stated on Hugging Face):** **MIT** — redistribution is allowed; retain the **MIT license and copyright notice** in your distribution if you ship GSM8K-derived files.

**Suggested citation:**

```bibtex
@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and others},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

---

### 3) GitHub pull requests — AbsenceBench (`github_prs`)

- **Role in the paper:** Code-diff / PR domain for AbsenceBench-style evaluation.
- **Typical source:** [harveyfin/AbsenceBench](https://huggingface.co/datasets/harveyfin/AbsenceBench), config `github_prs`; project repo [harvey-fin/absence-bench](https://github.com/harvey-fin/absence-bench).
- **License (as stated on Hugging Face dataset metadata):** **CC BY-SA 4.0** — attribution required; **Share-Alike**: if you **adapt** the data and share it publicly, adapted material typically must be licensed **CC BY-SA 4.0** as well. If you only load from Hugging Face at runtime and do not redistribute modified shards, your obligations differ from shipping a derived JSONL in this repo—plan accordingly.

**Suggested citation:**

```bibtex
@misc{fu2025absencebenchlanguagemodelscant,
  title={AbsenceBench: Language Models Can't Tell What's Missing},
  author={Anonymous},
  year={2025},
  eprint={2506.11440},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2506.11440},
}
```

---

### Practical checklist when bundling data with this code

1. **Legal / CUAD-QA:** Include attribution + CC BY 4.0 notice for any shipped CUAD-derived files.
2. **GSM8K:** Include MIT license text or required notices alongside shipped GSM8K files.
3. **AbsenceBench PR:** Include attribution + CC BY-SA 4.0 notice; if your shipped JSONL is a transformed subset of `github_prs`, treat it as an adaptation and comply with **Share-Alike** when you distribute it.
4. **Never commit API keys** or non-public snapshots that violate GitHub’s terms or the dataset licenses.

## Input Data Contract

The preprocessing pipeline expects JSONL with one document per line:

```json
{"id": "doc-1", "lines": ["...", "..."], "metadata": {...}}
```

- Required: `lines` (array of strings)
- Optional: `id`, `metadata`

## Generated Data Fields

`dataset_construction/process_data.py` writes processed documents with fields including:
- `_id`
- `original_context`
- `_modified_context`
- `_omitted_context`
- `omitted_index`
- `_metadata`
- `needles` and `needle_depths` (when using `--use_needle`)

## What to Include in Submission Archive

Recommended:
- code
- configs
- lightweight example data or scripts for data generation/download
- optional compliant snapshots of the three third-party inputs (only if you satisfy the licenses in **Third-party raw data** above)
- documentation (`README.md`, `REPRODUCIBILITY.md`, this file)

Do not include by default:
- provider API keys or `.env` files
- large experimental outputs (`data/`, `results/`, logs), unless required by venue artifact policy
- proprietary or restricted data

## Data Download Helper Scripts

- `scripts/download.sh`

Review URLs, terms, and expected file sizes before distributing artifacts.

# Transformations

This folder currently keeps the text-fragmentation implementation used by the paper:

- `spacing_transform.py`

Previously available transformation scripts (reverse/ROT/spelling/code transforms) were removed from the paper-facing artifact to keep the repository aligned with reported experiments.

## Supported Modes in `spacing_transform.py`

Main paper-relevant modes:

- `intra_word`: random-slot insertion within eligible character gaps (used for WFR sweeps)
- `custom_intra_word`: fixed-slot insertion with configurable slot/probability (used for fixed-slot control)

Other utility modes still available:

- `random_spaces`
- `tabs`
- `leading`
- `trailing`
- `mixed`
- `extreme`

## Example Commands

### A) Random-slot fragmentation (main U-curve pipeline input generation)

```bash
python transformations/spacing_transform.py \
  --input data/legal_docs.jsonl \
  --mode intra_word \
  --intensity 0.3 \
  --seed 42 \
  --output_dir data/transformed
```

### B) Fixed-slot fragmentation (first slot, probability p)

```bash
python transformations/spacing_transform.py \
  --input data/legal_docs.jsonl \
  --mode custom_intra_word \
  --gap_n 1 \
  --gap_p 0.3 \
  --seed 42 \
  --output_dir data/transformed
```

### C) Process multiple files using a glob

```bash
python transformations/spacing_transform.py \
  --input "data/*.jsonl" \
  --mode intra_word \
  --intensity 0.2 \
  --seed 42 \
  --output_dir data/transformed
```

## Key Arguments

- `--input`: single file path or glob pattern
- `--file_list`: optional text file listing input files (one per line)
- `--mode`: transformation mode
- `--intensity`: insertion probability for random modes
- `--gap_n`: target gap index (1-indexed) for `custom_intra_word`
- `--gap_m`: target gap quantile (0.0-1.0) for `custom_intra_word`
- `--gap_p`: insertion probability once a target slot is selected (for `custom_intra_word`)
- `--seed`: random seed
- `--output_dir`: output directory

## Notes

- For paper-facing runs, prefer `intra_word` and `custom_intra_word`.
- The script stores per-document `word_fragmentation_rate` metadata for `intra_word` / `custom_intra_word`.
- Keep transformation configuration and random seed in your experiment logs for reproducibility.

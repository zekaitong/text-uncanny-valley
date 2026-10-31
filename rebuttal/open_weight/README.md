# Open-weight experiments

`open_model_full_runner.py` runs one model per process across AbsenceBench,
NIAH, and GSM8K. The exact historical prompts are embedded in
`absence_prompt()` and `run_math_matrix()`. `legacy_evaluator/` is the evaluator
snapshot used by these runs; it is deliberately separate from the current
top-level evaluator. `legacy_evaluator/scorer.py` extracts only the frozen
scoring and document-loading functions, so running the open-weight code does
not import provider SDKs or read `.env`.

Checkpoint repository IDs and immutable revisions are listed in
`models.yaml`. By default the runner requires a populated Hugging Face cache.
Pass `--allow-download` to permit downloads, or `--model-path` to use a local
checkpoint directory.

Example:

```bash
python rebuttal/open_weight/open_model_full_runner.py \
  --model-key qwen2.5-7b-instruct \
  --tasks absencebench \
  --absence-domains ld \
  --rates ori,1,5,10 \
  --model-path /path/to/checkpoint
```

For Slurm, export `MODEL_KEY` and submit `submit_open_model_full.slurm`.
Inputs follow the paths documented in the runner and are not redistributed.


# Mechanistic probing

`mechanistic_probe_v2.py` measures prefix-position logit-lens recovery,
hidden-state divergence, and attention entropy across the eleven WFR levels.
The script records every resolved checkpoint and run parameter in
`probe_v2_run_config.json`.

Pinned public checkpoint IDs replace cluster-local cache paths. Set
`MODEL_PATH_<MODEL_KEY>` (non-alphanumeric characters become underscores) to
use a local snapshot, or set `HF_ALLOW_DOWNLOAD=1` to permit downloads.

```bash
MODEL_PATH_QWEN2_5_7B_INSTRUCT=/path/to/checkpoint \
python rebuttal/mechanistic_probing/mechanistic_probe_v2.py \
  --models qwen2.5-7b-instruct --domains ld --rates 0,0.5,1
```

No probe outputs are checked in. The script writes the resolved run
configuration, integrity table, per-example records, and aggregate summaries to
the selected output directory. Those files and the benchmark documents are
intentionally omitted and can be regenerated with authorized benchmark inputs.


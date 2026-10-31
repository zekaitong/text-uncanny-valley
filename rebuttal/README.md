# Rebuttal and camera-ready experiments

This directory releases the minimal code used for the additional experiments
described in the rebuttal and appendices. Raw benchmark documents,
provider responses, API keys, cluster logs, model weights, and per-request
Gemini payloads are intentionally excluded.

| Module | Contents | Priority |
|---|---|---|
| `open_weight/` | Open-weight AbsenceBench, NIAH, and GSM8K runner; pinned model revisions; prompts; Slurm launcher; exact legacy evaluator snapshot | P0 |
| `dictionary_repair/` | External-dictionary spacing repair, dictionary snapshot, and example config | P0 |
| `gemini_legal_11point/` | Ordinary Gemini API workflow used to complete the eight missing rates in the final 11-point Legal sweep | P0 |
| `mechanistic_probing/` | Mechanistic probe v2 and pinned checkpoint/parameter definitions | P1 |

No experimental outputs, result datasets, generated tables, or generated
figures are checked in. Re-running inference may produce different outputs
when model serving software or preview API aliases change.

The Gemini package intentionally omits legal source documents, per-request
payloads, raw provider responses, omission mappings, and its generated execution
manifest. Authorized users create that ignored manifest locally with
`prepare_execution.py`.

Install the main requirements plus the optional packages in
`requirements-rebuttal.txt`. GPU memory and model-specific dependencies vary;
the 70B AWQ checkpoint requires a compatible quantization stack.


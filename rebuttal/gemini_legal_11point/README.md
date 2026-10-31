# Gemini Legal 11-point completion

This is ordinary Gemini API inference, not Google Batch. It prepares 100
requests at each missing WFR level (.1/.2/.3/.4/.6/.7/.8/.9) and combines
them with an authorized historical inventory. Model: gemini-3-flash-preview.
Protocol: F22 Option A.

Both displayed documents and scoring references are repaired. Inputs, omission
indices, prompt and legacy scorer are frozen and hashed. No new output cleaning,
new parser, reference normalization, dictionary change or model substitution.
Historical generation options are retained: thinkingBudget=0; temperature,
max output tokens and stop sequences omitted. The 10-minute transport timeout
and single HTTP attempt are execution safeguards, not decoding changes.

## Paths and environment

Run from `rebuttal/gemini_legal_11point`. Set `TEXT_UNCANNY_PROJECT_ROOT` when
the repository root cannot be inferred, and optionally set `PYTHON_BIN` in the
Slurm launcher. The historical package used google-genai 1.63.0.
Offline preparation does not read credentials or construct an API client.
No job is submitted by any Python file in this package.

## Manual Initial Submission

Only run after confirming collaborator spending permission. In your SSH terminal,
GEMINI_API_KEY must already be exported; do not paste it in chat or put it in files.
Slurm inherits it via --export=ALL. Do not enable shell tracing or print your env.

```bash
cd rebuttal/gemini_legal_11point
python3 prepare_execution.py
python3 manage.py validate
sbatch --export=ALL,PLAN_ID=initial standard4.slurm
```

The array has eight WFR tasks, at most four simultaneous workers (`0-7%4`).
Each task sends its 100 document requests serially. Four shared file-lock slots
provide an additional cross-process concurrency guard. Inference requires --run
and Slurm array variables and rejects login-node execution. Each plan binds to
one array job ID; do not submit initial a second time. Queueing is not included
in the roughly 32-minute service estimate; allow 30-60 minutes service, possibly
1-2 hours with API variability. The allocation is 2 hours/task, 1 CPU, 2 GB,
partition general, no GPU. Check current provider pricing and shared API limits
before submission.

You may close this chat or disconnect SSH after sbatch accepts the array.
Slurm continues execution and writes logs and per-request files on the cluster.

## Progress and Offline Collection

```bash
squeue -u "$USER"
python3 manage.py status
python3 manage.py collect
```

`logs/<array>_<task>.out/.err` show task activity. `attempts/<key>/0001.json`
is created BEFORE sending. Raw responses, usage, model version and finish reasons
are saved immediately after receiving, before text parsing/scoring. State writes
are atomic and fsynced. Never delete or edit an attempt file to rerun a request.

Any API exception stops further requests in that plan via a shared STOP flag;
already in-flight requests can finish and save. No SDK/application auto retry.
An interrupted STARTED_UNCERTAIN record may already have incurred a charge.
Do not assume it is free or automatically rerun it. Exception messages and HTTP
headers are not stored, because they may contain sensitive request information.

Each collect creates a NEW dated `derived/` snapshot: summary, curves CSV,
per-document scores, paired differences, new raw-response mapping/usage and
missing/error table. No original scores are overwritten and absent responses
are not zero-filled. The FIRST saved response for a key is used, never the best
score. Truncated, blocked or no-text returns are flagged separately; the legacy
scorer's treatment of returned None as empty is retained, not silently changed.
Saved-response counts must not be described as 100% valid model output without
checking those flags. API dates/defaults/preview checkpoint can drift; this is a
cross-date descriptive pipeline comparison, not a contemporaneous causal study.

## Safe Manual Resume

Wait until every task of the previous array has left squeue. Run status/collect
first. The resume command queries Slurm read-only and refuses if a known array
is still active. Default resume includes ONLY never-attempted keys:

```bash
python3 manage.py resume --name resume1
python3 manage.py resume --name resume1 --write
sbatch --export=ALL,PLAN_ID=resume1 standard4.slurm
```

The first command is dry-run. No submit happens until you invoke sbatch.
Resume plans create new attempt numbers without replacing old records. Keys
that already have a saved API response are never regenerated: offline collection
can retry parsing/scoring without further inference. Plans may contain fewer
than eight nonempty WFR tasks, but retain the same array layout and four-slot cap.

To explicitly repeat API-error keys without a raw response, after investigating
quota/auth/config issues and accepting possible additional billing:

```bash
python3 manage.py resume --name retry1 --include-failed --acknowledge-possible-billing
```

Review this dry-run before adding --write and manually submitting PLAN_ID=retry1.
STARTED_UNCERTAIN keys are excluded even by --include-failed. Approve each such
key explicitly with `--retry-uncertain <key>` and the billing acknowledgment.
This package cannot determine whether an ambiguous failed attempt was billed.
Do not remove STOP files or rerun an existing plan to bypass these safeguards.

## Offline Validation Boundary

The public repository contains the workflow and portable legacy helper, but
deliberately contains neither an execution manifest nor request payloads,
per-document legal texts, omission mappings, raw provider responses, result
curves, or credentials. Authorized users run `prepare_execution.py` to copy
verified private inputs and generate the ignored `execution_manifest.json`.
Preparation, validation, status, and collection are CPU-only and can run on the
login node. `worker.py --run` cannot.

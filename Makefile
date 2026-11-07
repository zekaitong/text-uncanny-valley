-include .env

_conda = conda

SBATCH = $(_conda) run -n ${CONDA_ENV} sbatch

ifeq ($(strip $(CONDA_ENV)),)
	PYTHON_CMD = python
else
	PYTHON_CMD = $(_conda) run -n ${CONDA_ENV} python
endif

CONFIG_FILE ?= configs/config.yaml
DEPENDENCY ?=
SWEEP_ROOT ?=
NIAH_ROOT ?= results/NIAH
NIAH_SUFFIX ?= .fixed2
NIAH_SUMMARY_DIR ?= summaries_fixed2
NIAH_RESULT_FILE ?= results.fixed2.json
GSM_MODELS ?= gemini-3-flash-preview gpt-5.2 claude-haiku-4-5
GSM_SAMPLE_SIZE ?= 500
GSM_BATCH_SIZE ?= 5
GSM_SEED ?= 42
GSM_PARSE_MODE ?= strict

# Construct dependency flag only if specified
ifneq ($(strip $(DEPENDENCY)),)
    DEP_FLAG = --dependency=$(DEPENDENCY)
endif

.PHONY: run_benchmark run_and_summarize gsm_reasoning_strict niah_fix offline_test offline_test_full
run_benchmark:
	$(SBATCH) \
		--parsable \
		--partition=$(SLURM_PARTITION) \
		$(DEP_FLAG) \
		--export=ALL,CONFIG_FILE=$(CONFIG_FILE) \
		submit_benchmark.slurm

run_and_summarize:
	@if [ -z "$(SWEEP_ROOT)" ]; then \
		echo "SWEEP_ROOT is required. Example:"; \
		echo "make run_and_summarize CONFIG_FILE=configs/xxx.yaml SWEEP_ROOT=data/ld_seed42"; \
		exit 1; \
	fi
	./scripts/submit_and_summarize.sh "$(CONFIG_FILE)" "$(SWEEP_ROOT)" "$(SLURM_PARTITION)"

# Standalone Experiment C (GSM8K reasoning) with strict answer parsing.
gsm_reasoning_strict:
	$(PYTHON_CMD) gsm8k_test/math_test.py \
		--models $(GSM_MODELS) \
		--sample_size $(GSM_SAMPLE_SIZE) \
		--batch_size $(GSM_BATCH_SIZE) \
		--seed $(GSM_SEED) \
		--parse_mode $(GSM_PARSE_MODE)

# Recalculate NIAH metrics from saved responses and rebuild summaries.
niah_fix:
	$(PYTHON_CMD) summaries/recompute_niah_scores.py "$(NIAH_ROOT)" --output_suffix "$(NIAH_SUFFIX)"
	$(PYTHON_CMD) summaries/build_niah_summary_from_results.py "$(NIAH_ROOT)" \
		--result_file_name "$(NIAH_RESULT_FILE)" \
		--output_dir "$(NIAH_SUMMARY_DIR)"

# Fully offline regression checks (no external API calls).
offline_test:
	$(PYTHON_CMD) scripts/offline_regression.py

# Broader offline integration checks across non-API CLIs.
offline_test_full:
	$(PYTHON_CMD) scripts/offline_full_check.py

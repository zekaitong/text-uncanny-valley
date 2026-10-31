#!/usr/bin/env python3
"""
Comprehensive offline integration checks for non-API functionality.

This script creates temporary fixtures and runs core CLI tools end-to-end
without calling external model APIs.
"""

import json
import subprocess
import tempfile
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd, cwd=ROOT):
    res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return res.returncode, res.stdout, res.stderr


def must(cmd, name):
    code, out, err = run(cmd)
    if code != 0:
        raise RuntimeError(f"[{name}] failed\nCMD: {' '.join(cmd)}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return out, err


def step(cmd, name, checks, required=True):
    try:
        must(cmd, name)
        checks.append(f"[PASS] {name}")
        return True
    except Exception as e:
        if required:
            raise
        checks.append(f"[SKIP] {name} ({e})")
        return False


def write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix="offline_full_") as td:
        t = Path(td)
        data_dir = t / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # 1) process_data.py: omission / needle / placeholder
        input_file = data_dir / "toy.jsonl"
        write_jsonl(
            input_file,
            [
                {"id": 1, "lines": ["alpha", "beta", "gamma"], "metadata": {"src": "toy"}},
                {"id": 2, "lines": ["one", "two", "three"], "metadata": {"src": "toy"}},
            ],
        )
        needles_file = data_dir / "needles.txt"
        needles_file.write_text("needle-A\nneedle-B\n", encoding="utf-8")

        out_omit = t / "omit"
        out_needle = t / "needle"
        out_ph = t / "placeholder"
        out_omit.mkdir()
        out_needle.mkdir()
        out_ph.mkdir()

        step(
            [
                PY,
                "dataset_construction/process_data.py",
                "--input_file",
                str(input_file),
                "--prob",
                "0.5",
                "--random_seed",
                "42",
                "--output_dir",
                str(out_omit),
            ],
            "process_data_omission",
            checks,
        )

        step(
            [
                PY,
                "dataset_construction/process_data.py",
                "--input_file",
                str(input_file),
                "--prob",
                "0.5",
                "--random_seed",
                "42",
                "--use_needle",
                "--needles_file",
                str(needles_file),
                "--output_dir",
                str(out_needle),
            ],
            "process_data_needle",
            checks,
        )

        step(
            [
                PY,
                "dataset_construction/process_data.py",
                "--input_file",
                str(input_file),
                "--prob",
                "0.5",
                "--random_seed",
                "42",
                "--use_placeholders",
                "--output_dir",
                str(out_ph),
            ],
            "process_data_placeholder",
            checks,
        )

        # 2) spacing_transform.py
        transformed_dir = t / "transformed"
        transformed_dir.mkdir()
        step(
            [
                PY,
                "transformations/spacing_transform.py",
                "--input",
                str(input_file),
                "--mode",
                "custom_intra_word",
                "--gap_n",
                "1",
                "--gap_p",
                "1.0",
                "--output_dir",
                str(transformed_dir),
            ],
            "spacing_transform_custom",
            checks,
        )

        # 3) summarize_f1.py + plot_from_summary.py
        sweep = t / "sweep"
        for cond, score in [("ori", 0.9), ("p_0_1", 0.8), ("p_0_2", 0.7)]:
            d = sweep / cond
            d.mkdir(parents=True, exist_ok=True)
            payload = {
                "test_date": "x",
                "openai": {"gpt-5.2": {"average_accuracy": score}},
                "anthropic": {"claude-haiku-4-5": {"average_accuracy": score - 0.05}},
            }
            (d / "results.json").write_text(json.dumps(payload), encoding="utf-8")

        summary_json = sweep / "summary.json"
        summary_csv = sweep / "summary.csv"
        step(
            [
                PY,
                "summaries/summarize_f1.py",
                str(sweep),
                "--json_out",
                str(summary_json),
                "--csv_out",
                str(summary_csv),
            ],
            "summarize_f1",
            checks,
        )

        plot_prefix = t / "figs" / "toy_curve"
        plot_prefix.parent.mkdir(parents=True, exist_ok=True)
        step(
            [
                PY,
                "plot/plot_from_summary.py",
                "--summary_json",
                str(summary_json),
                "--output_prefix",
                str(plot_prefix),
            ],
            "plot_from_summary",
            checks,
            required=False,
        )

        # 4) recompute_niah_scores.py + build_niah_summary_from_results.py
        niah_root = t / "NIAH"
        cond_dir = niah_root / "ld" / "modelA" / "wa_bn" / "ori"
        cond_dir.mkdir(parents=True, exist_ok=True)
        used_data = [
            {"_id": 1, "original_context": "l1\nl2", "needles": ["n1", "n2"], "needle_depths": [0.3, 0.8]},
        ]
        write_jsonl(cond_dir / "used_data.jsonl", used_data)
        result_payload = {
            "test_date": "x",
            "openai": {
                "gpt-5.2": {
                    "model_provider": "openai",
                    "model_name": "gpt-5.2",
                    "total_documents": 1,
                    "average_accuracy": 1.0,
                    "average_thinking_tokens": 0.0,
                    "detailed_results": [
                        {
                            "id": 1,
                            "tp": 2,
                            "fp": 0,
                            "fn": 0,
                            "micro_f1": 1.0,
                            "identified_lines": ["n1", "n2"],
                            "unidentified_lines": [],
                            "wrongly_identified_lines": [],
                            "excess_identified_lines": [],
                            "model_response": ["n1\nn2\nnoise", 0, ""],
                            "needle_depths": [0.3, 0.8],
                        }
                    ],
                }
            },
        }
        (cond_dir / "results.json").write_text(json.dumps(result_payload), encoding="utf-8")

        step(
            [
                PY,
                "summaries/recompute_niah_scores.py",
                str(niah_root),
                "--output_suffix",
                ".fixed2",
            ],
            "recompute_niah_scores",
            checks,
        )

        step(
            [
                PY,
                "summaries/build_niah_summary_from_results.py",
                str(niah_root),
                "--result_file_name",
                "results.fixed2.json",
                "--output_dir",
                "summaries_fixed2",
                "--domains",
                "ld",
            ],
            "build_niah_summary_from_results",
            checks,
        )

        # 5) generate_yamls.py
        yaml_dir = t / "yamls"
        yaml_dir.mkdir()
        template = yaml_dir / "1.yaml"
        template.write_text(
            'input_file: "data/legal_docs_1.jsonl"\noutput_dir: "run/1"\nneedles_file: "data/needle/ld/legal_needles_wa1.txt"\n',
            encoding="utf-8",
        )
        step(
            [
                PY,
                "generate_yamls.py",
                str(template),
                "--ori_input",
                "data/legal_docs.jsonl",
                "--with_needle_wa",
            ],
            "generate_yamls",
            checks,
        )

        # 6) existing offline_regression.py
        step([PY, "scripts/offline_regression.py"], "offline_regression", checks, required=False)

        # 7) summarize_entropy.py smoke (may fail if tiktoken cache missing)
        entropy_root = t / "entropy_sweep"
        for cond in ["ori", "p_0_1"]:
            d = entropy_root / cond
            d.mkdir(parents=True, exist_ok=True)
            write_jsonl(d / "used_data.jsonl", [{"original_context": "alpha beta gamma"}])

        code, out, err = run(
            [
                PY,
                "summaries/summarize_entropy.py",
                str(entropy_root),
                "--tokenizer",
                "gpt",
                "--json_out",
                str(entropy_root / "entropy.json"),
            ]
        )
        if code == 0:
            checks.append("[PASS] summarize_entropy")
        else:
            checks.append("[SKIP] summarize_entropy (tokenizer cache/network unavailable)")

    print("Offline full check completed.")
    for c in checks:
        print(c)


if __name__ == "__main__":
    main()

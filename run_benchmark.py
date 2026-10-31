#!/usr/bin/env python3
"""
run_benchmark.py
End-to-end script to process data and run evaluation on custom domains
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import yaml


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")
    # Print full command for easier debugging
    print(f"Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {description} failed!")
        print(f"Exit code: {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Given a dataset of documents, creates omitted versions and runs AbsenceBench eval on it"
    )

    parser.add_argument("--input_file", type=str, help="Path to the input file")
    parser.add_argument("-p", "--prob", type=float, default=0.1)
    parser.add_argument("-d", "--delimiter", type=str, default="\n")
    parser.add_argument("-m", "--max_lines", type=int)
    parser.add_argument("--document_type", type=str, default="document")
    parser.add_argument("--use_needle", action="store_true")
    parser.add_argument("--use_placeholders", action="store_true")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--processed_input", action="store_true")

    # evaluation stage arguments
    parser.add_argument("--sample_size", type=int)
    parser.add_argument("--provider_models", type=str, nargs="+", default=["openai:gpt-4.1-nano-2025-04-14"])
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--system_prompt", type=str)
    parser.add_argument("--reasoning_effort", type=str, choices=["low", "medium", "high"])
    parser.add_argument("--user_message", type=str)

    parser.add_argument("--needles_file", type=str, help="Path to needles file")

    # Added icl_file support so YAML config can pass it through
    parser.add_argument("--icl_file", type=str, help="Path to ICL examples file")

    # other
    parser.add_argument("--results_handle", type=str)
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    # Load config from YAML
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        valid_keys = {a.dest for a in parser._actions}
        for k, v in cfg.items():
            if k in valid_keys and v is not None:
                args.__dict__[k] = v

    if not args.input_file:
        parser.error("--input_file is required")

    if not Path(args.input_file).exists():
        print(f"❌ Error: Input file '{args.input_file}' does not exist!")
        sys.exit(1)

    # Keep existing behavior: force run outputs under data/
    if args.output_dir:
        run_dir = f"data/{args.output_dir}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"data/{args.document_type}_{timestamp}"

    os.makedirs(run_dir, exist_ok=True)
    processed_data = f"{run_dir}/processed_data.jsonl"
    results_file = f"{run_dir}/{args.results_handle}.json" if args.results_handle else f"{run_dir}/results.json"

    print("\n" + "=" * 60)
    print("ABSENCE BENCHMARK - END-TO-END PIPELINE")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Document type: {args.document_type}")

    # ========================================
    # STEP 1: Process Data
    # ========================================
    if not args.processed_input:
        process_cmd = [
            "python", "dataset_construction/process_data.py",
            "--input_file", args.input_file,
            "-p", str(args.prob),
            "-d", args.delimiter,
            "--document_type", args.document_type,
            "--random_seed", str(args.random_seed),
            "--output_dir", run_dir,
        ]
        if args.max_lines: process_cmd.extend(["-m", str(args.max_lines)])
        if args.use_placeholders:
            process_cmd.append("--use_placeholders")

        if args.use_needle:
            process_cmd.append("--use_needle")
            if args.needles_file:
                process_cmd.extend(["--needles_file", args.needles_file])

        if not run_command(process_cmd, "STEP 1: Processing data"):
            sys.exit(1)

    # ========================================
    # STEP 2: Run Evaluation
    # ========================================
    # Use robust list-based command construction and explicit argument forwarding
    test_cmd = [
        "python", "tests/test_llms.py",
        "--input_file", processed_data,
        "--output", results_file,
        "--document_type", args.document_type,
        "--delimiter", args.delimiter,
        "--batch_size", str(args.batch_size),
        "--random_seed", str(args.random_seed),
    ]

    # Add multi-value flag
    test_cmd.append("--provider_models")
    test_cmd.extend(args.provider_models)

    if args.sample_size:
        test_cmd.extend(["--sample_size", str(args.sample_size)])
    if args.use_needle:
        test_cmd.append("--use_needle")
    if args.thinking:
        test_cmd.append("--thinking")
    
    # Explicitly pass ICL file path when provided
    if hasattr(args, 'icl_file') and args.icl_file:
        test_cmd.extend(["--icl_file", args.icl_file])

    if args.system_prompt:
        test_cmd.extend(["--system_prompt", args.system_prompt])
    if args.user_message:
        test_cmd.extend(["--user_message", args.user_message])
    if args.reasoning_effort:
        test_cmd.extend(["--reasoning_effort", args.reasoning_effort])

    success = run_command(test_cmd, "STEP 2: Running evaluation")

    if not success:
        print("\n❌ Pipeline failed at evaluation step")
        sys.exit(1)

    print(f"\n✅ Evaluation complete: {results_file}")

if __name__ == "__main__":
    main()
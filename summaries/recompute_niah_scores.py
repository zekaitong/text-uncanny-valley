#!/usr/bin/env python3
"""
Recompute NIAH detailed scores from existing results.json + used_data.jsonl.

Fixes two issues in legacy NIAH scoring:
1) strict line-level needle matching (no substring matching)
2) count hallucinated non-needle lines as FP
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple


def split_lines(text: str, delimiter: str = "\n") -> List[str]:
    if not text:
        return []
    return [line.strip() for line in str(text).split(delimiter) if line.strip()]


def get_response_text(model_response: Any) -> str:
    if isinstance(model_response, list):
        if model_response and model_response[0] is not None:
            return str(model_response[0])
        return ""
    if isinstance(model_response, str):
        return model_response
    return ""


def recompute_niah_for_doc(document: Dict[str, Any], response_text: str, delimiter: str = "\n") -> Dict[str, Any]:
    original_lines = split_lines(document.get("original_context", ""), delimiter)
    needles_raw = [n for n in document.get("needles", []) if str(n).strip()]

    response_lines = split_lines(response_text, delimiter)
    response_lines_lower = [line.lower() for line in response_lines]
    original_lines_lower = set(line.lower() for line in original_lines)
    needle_counter = Counter(str(n).strip().lower() for n in needles_raw)
    response_counter = Counter(response_lines_lower)

    results = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "identified_lines": [],
        "unidentified_lines": [],
        "wrongly_identified_lines": [],
        "excess_identified_lines": [],
    }

    matched_needles = Counter()
    for needle_line, needle_count in needle_counter.items():
        matched_count = min(needle_count, response_counter.get(needle_line, 0))
        if matched_count > 0:
            results["tp"] += matched_count
            matched_needles[needle_line] += matched_count
            for _ in range(matched_count):
                results["identified_lines"].append(needle_line)

        missed_count = needle_count - matched_count
        if missed_count > 0:
            results["fn"] += missed_count
            for _ in range(missed_count):
                results["unidentified_lines"].append(needle_line)

    # Any unconsumed output line is FP.
    for original_line in response_lines:
        line = original_line.lower()
        if matched_needles.get(line, 0) > 0:
            matched_needles[line] -= 1
            continue
        results["fp"] += 1
        results["wrongly_identified_lines"].append(original_line)
        if line not in original_lines_lower:
            results["excess_identified_lines"].append(original_line)

    denom = 2 * results["tp"] + results["fp"] + results["fn"]
    results["micro_f1"] = (2 * results["tp"] / denom) if denom > 0 else 0.0
    results["needle_depths"] = document.get("needle_depths", [])
    return results


def load_used_data(path: Path) -> Dict[str, Dict[str, Any]]:
    docs: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id = obj.get("_id", obj.get("id"))
            docs[str(doc_id)] = obj
    return docs


def process_results_file(results_path: Path, in_place: bool, output_suffix: str, delimiter: str) -> Tuple[float, float]:
    used_data_path = results_path.parent / "used_data.jsonl"
    if not used_data_path.exists():
        raise FileNotFoundError(f"Missing used_data.jsonl next to: {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    docs = load_used_data(used_data_path)
    providers = [k for k in payload.keys() if k != "test_date"]

    total_old = 0.0
    total_new = 0.0
    count_models = 0

    for provider in providers:
        for model_name, model_block in payload[provider].items():
            detailed = model_block.get("detailed_results", [])
            if not detailed:
                continue

            new_detailed = []
            for row in detailed:
                doc_id = str(row.get("id"))
                document = docs.get(doc_id)
                if document is None:
                    # Keep row untouched if doc can't be found.
                    new_detailed.append(row)
                    continue

                response_text = get_response_text(row.get("model_response"))
                new_metrics = recompute_niah_for_doc(document, response_text, delimiter=delimiter)

                updated = dict(row)
                updated["tp"] = new_metrics["tp"]
                updated["fp"] = new_metrics["fp"]
                updated["fn"] = new_metrics["fn"]
                updated["identified_lines"] = new_metrics["identified_lines"]
                updated["unidentified_lines"] = new_metrics["unidentified_lines"]
                updated["wrongly_identified_lines"] = new_metrics["wrongly_identified_lines"]
                updated["excess_identified_lines"] = new_metrics["excess_identified_lines"]
                updated["micro_f1"] = new_metrics["micro_f1"]
                updated["needle_depths"] = new_metrics["needle_depths"]
                new_detailed.append(updated)

            old_avg = model_block.get("average_accuracy", 0.0)
            new_avg = sum(r.get("micro_f1", 0.0) for r in new_detailed) / len(new_detailed)
            model_block["detailed_results"] = new_detailed
            model_block["average_accuracy"] = new_avg

            total_old += old_avg
            total_new += new_avg
            count_models += 1

    if in_place:
        output_path = results_path
    else:
        output_path = results_path.with_name(results_path.stem + output_suffix + results_path.suffix)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    avg_old = total_old / count_models if count_models else 0.0
    avg_new = total_new / count_models if count_models else 0.0
    return avg_old, avg_new


def main():
    parser = argparse.ArgumentParser(description="Recompute NIAH FP and Micro-F1 from existing result artifacts.")
    parser.add_argument("target", help="Path to a NIAH results.json file or to a directory containing many of them")
    parser.add_argument("--in_place", action="store_true", help="Overwrite original results.json files")
    parser.add_argument("--output_suffix", default=".corrected", help="Suffix for corrected files when not in-place")
    parser.add_argument("--delimiter", default="\n", help="Line delimiter used in prompts/data (default: newline)")
    args = parser.parse_args()

    target = Path(args.target)
    if not target.exists():
        raise FileNotFoundError(f"Target not found: {target}")

    delimiter = args.delimiter.encode("utf-8").decode("unicode_escape")

    if target.is_file():
        result_files = [target]
    else:
        result_files = sorted(target.glob("**/results.json"))

    if not result_files:
        raise RuntimeError(f"No results.json found under: {target}")

    processed = 0
    old_means = []
    new_means = []
    for rf in result_files:
        try:
            old_avg, new_avg = process_results_file(
                rf,
                in_place=args.in_place,
                output_suffix=args.output_suffix,
                delimiter=delimiter,
            )
            print(f"[ok] {rf} | avg_accuracy: {old_avg:.6f} -> {new_avg:.6f}")
            processed += 1
            old_means.append(old_avg)
            new_means.append(new_avg)
        except Exception as e:
            print(f"[skip] {rf} | {e}")

    if processed == 0:
        raise RuntimeError("No files were processed successfully.")

    overall_old = sum(old_means) / len(old_means)
    overall_new = sum(new_means) / len(new_means)
    print("=" * 80)
    print(f"Processed files: {processed}")
    print(f"Overall mean avg_accuracy: {overall_old:.6f} -> {overall_new:.6f}")
    print(f"Delta (old - new): {overall_old - overall_new:.6f}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import os
import re
from typing import Dict, List, Tuple


def parse_rate(name: str):
    """
    Parse rate from folder names like:
    - ori
    - p_0_0 / p_0_1 / p_1_0
    - 0.0 / 0.1 / 1.0
    """
    low = name.lower()
    if low == "ori":
        return 0.0

    m = re.match(r"^p_(\d+)_(\d+)$", low)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")

    try:
        return float(name)
    except ValueError:
        return None


def parse_results_json(path: str) -> Dict[str, float]:
    """Return {provider:model -> average_accuracy} from results.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, float] = {}
    for provider, provider_payload in data.items():
        if provider == "test_date" or not isinstance(provider_payload, dict):
            continue
        for model_name, model_payload in provider_payload.items():
            if (
                isinstance(model_payload, dict)
                and "average_accuracy" in model_payload
                and model_payload["average_accuracy"] is not None
            ):
                key = f"{provider}:{model_name}"
                out[key] = float(model_payload["average_accuracy"])
    return out


def collect(root: str) -> Tuple[List[float], Dict[str, List[float]], List[str], List[str], List[str]]:
    """Collect rate-indexed per-model Micro-F1 from condition folders."""
    by_rate: Dict[float, Dict[str, float]] = {}
    all_models = set()
    skipped_unparseable: List[str] = []
    skipped_missing_results: List[str] = []
    skipped_bad_results: List[str] = []

    for subdir in os.listdir(root):
        subdir_path = os.path.join(root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        rate = parse_rate(subdir)
        if rate is None:
            skipped_unparseable.append(subdir)
            continue

        results_file = os.path.join(subdir_path, "results.json")
        if not os.path.exists(results_file):
            skipped_missing_results.append(subdir)
            continue
        try:
            model_scores = parse_results_json(results_file)
        except Exception:
            skipped_bad_results.append(subdir)
            continue

        by_rate[rate] = model_scores
        all_models.update(model_scores.keys())

    rates = sorted(by_rate.keys())
    model_to_series: Dict[str, List[float]] = {}
    for model in sorted(all_models):
        series = []
        for rate in rates:
            series.append(by_rate[rate].get(model))
        model_to_series[model] = series
    return rates, model_to_series, skipped_unparseable, skipped_missing_results, skipped_bad_results


def print_vertical(rates: List[float], model_to_series: Dict[str, List[float]]):
    """Legacy-friendly vertical output."""
    for r in rates:
        print(r)
    for model in sorted(model_to_series.keys()):
        for v in model_to_series[model]:
            print(v if v is not None else "Missing")


def write_csv(path: str, rates: List[float], model_to_series: Dict[str, List[float]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    models = sorted(model_to_series.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rate"] + models)
        for i, rate in enumerate(rates):
            row = [rate]
            for m in models:
                row.append(model_to_series[m][i])
            writer.writerow(row)


def write_json(path: str, rates: List[float], model_to_series: Dict[str, List[float]]):
    payload = {"rates": rates, "series": model_to_series}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize Micro-F1 across fragmentation-rate run folders."
    )
    parser.add_argument("path", help="Root folder containing per-rate subfolders")
    parser.add_argument(
        "--format",
        choices=["vertical", "table"],
        default="table",
        help="Output format for stdout",
    )
    parser.add_argument("--csv_out", type=str, help="Optional CSV output path")
    parser.add_argument("--json_out", type=str, help="Optional JSON output path")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise FileNotFoundError(f"Path not found: {args.path}")

    rates, model_to_series, skipped_unparseable, skipped_missing_results, skipped_bad_results = collect(args.path)

    if skipped_unparseable:
        print(
            "Warning: skipped folders with unparseable rate names: "
            + ", ".join(sorted(skipped_unparseable))
        )
        print(
            "Expected folder naming examples: ori, p_0_1, p_0_2, ... p_1_0, or numeric names like 0.1"
        )
    if skipped_missing_results:
        print(
            "Warning: skipped folders without results.json: "
            + ", ".join(sorted(skipped_missing_results))
        )
    if skipped_bad_results:
        print(
            "Warning: skipped folders with unreadable results.json: "
            + ", ".join(sorted(skipped_bad_results))
        )

    if not rates:
        raise RuntimeError(
            f"No usable results found under: {args.path}\n"
            "Make sure each WFR condition is saved in a parseable subfolder name "
            "(e.g., ori, p_0_1, ..., p_1_0) and contains results.json."
        )

    if args.format == "vertical":
        print_vertical(rates, model_to_series)
    else:
        header = ["rate"] + sorted(model_to_series.keys())
        print("\t".join(header))
        for i, rate in enumerate(rates):
            row = [str(rate)]
            for m in sorted(model_to_series.keys()):
                value = model_to_series[m][i]
                row.append("" if value is None else f"{value:.6f}")
            print("\t".join(row))

    if args.csv_out:
        write_csv(args.csv_out, rates, model_to_series)
    if args.json_out:
        write_json(args.json_out, rates, model_to_series)

#!/usr/bin/env python3
"""
Build per-domain summary.json files for NIAH plotting from condition folders.

Expected layout:
  <root>/<domain>/<model>/wa_bn/<cond>/<result_file>
where cond is "ori" or numeric strings like "1".."10".
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def cond_to_rate(name: str) -> float:
    if name == "ori":
        return 0.0
    return float(name) / 10.0


def build_domain_summary(domain_dir: Path, result_file_name: str) -> Dict:
    model_series: Dict[str, Dict[float, float]] = {}
    all_rates = set()

    for model_dir in sorted([p for p in domain_dir.iterdir() if p.is_dir()]):
        wa_dir = model_dir / "wa_bn"
        if not wa_dir.exists():
            continue

        label = None
        points: Dict[float, float] = {}
        for cond_dir in sorted([p for p in wa_dir.iterdir() if p.is_dir()]):
            cond = cond_dir.name
            if cond != "ori" and not cond.isdigit():
                continue
            result_path = cond_dir / result_file_name
            if not result_path.exists():
                continue

            with open(result_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            providers = [k for k in payload.keys() if k != "test_date"]
            if not providers:
                continue
            provider = providers[0]
            models = list(payload[provider].keys())
            if not models:
                continue
            model = models[0]
            label = f"{provider}:{model}"

            score = payload[provider][model].get("average_accuracy")
            points[cond_to_rate(cond)] = score
            all_rates.add(cond_to_rate(cond))

        if label and points:
            model_series[label] = points

    rates = sorted(all_rates)
    series = {label: [pts.get(r) for r in rates] for label, pts in sorted(model_series.items())}
    return {"rates": rates, "series": series}


def main():
    parser = argparse.ArgumentParser(description="Build NIAH summary JSONs from per-condition result files.")
    parser.add_argument("root", help="Root NIAH directory, e.g. results/NIAH")
    parser.add_argument(
        "--result_file_name",
        default="results.fixed2.json",
        help="Result filename inside each condition directory (default: results.fixed2.json)",
    )
    parser.add_argument(
        "--output_dir",
        default="summaries_fixed2",
        help="Output folder under root (default: summaries_fixed2)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["ld", "gp", "gsm"],
        help="Domain directories to process (default: ld gp gsm)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for domain in args.domains:
        domain_dir = root / domain
        if not domain_dir.exists():
            print(f"[skip] missing domain: {domain_dir}")
            continue

        payload = build_domain_summary(domain_dir, args.result_file_name)
        out_path = out_dir / f"{domain}_niah_summary.{args.result_file_name.replace('.json', '')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()

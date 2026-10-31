import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from helpers import iter_series_for_paper_plot, load_summary


def main():
    parser = argparse.ArgumentParser(
        description="Plot line chart from summarize_f1 JSON output."
    )
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--title", default="Micro-F1 vs Fragmentation Rate")
    parser.add_argument("--ylabel", default="Micro-F1")
    parser.add_argument("--xlabel", default="Word Fragmentation Rate")
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument(
        "--keep_provider_prefix",
        action="store_true",
        help="Keep provider:model labels in legend.",
    )
    args = parser.parse_args()

    rates, series = load_summary(args.summary_json)
    rates = np.array(rates, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    strip = not args.keep_provider_prefix
    for model, label, color in iter_series_for_paper_plot(series, strip_provider=strip):
        y = np.array([np.nan if v is None else float(v) for v in series[model]], dtype=float)
        ax.plot(
            rates,
            y,
            label=label,
            color=color,
            linewidth=2,
            marker="o",
            markersize=2,
            markerfacecolor=color,
            markeredgewidth=0,
        )

    ax.set_title(args.title, fontsize=12, fontweight="bold")
    ax.set_xlabel(args.xlabel, fontsize=11)
    ax.set_ylabel(args.ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.9)

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{args.output_prefix}.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output_prefix}.pdf/.png")


if __name__ == "__main__":
    main()

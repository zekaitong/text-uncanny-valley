import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from helpers import iter_series_for_paper_plot, load_summary, save, style_axes, to_array


def main():
    parser = argparse.ArgumentParser(description="Plot U-curve with tokenization entropy overlay.")
    parser.add_argument("--seed_summaries", nargs="+", required=True, help="Summary JSON paths for multiple seeds")
    parser.add_argument("--entropy_json", required=True, help="Entropy JSON from summaries/summarize_entropy.py --json_out")
    parser.add_argument("--title", default="Github PRs")
    parser.add_argument("--output_prefix", default="uncanny_valley_gp_entropy")
    parser.add_argument("--keep_provider_prefix", action="store_true")
    args = parser.parse_args()

    per_seed = []
    rates = None
    for p in args.seed_summaries:
        r, series = load_summary(p)
        if rates is None:
            rates = r
        per_seed.append(series)
    rates_arr = np.array(rates, dtype=float)

    with open(args.entropy_json, "r", encoding="utf-8") as f:
        ent_payload = json.load(f)
    entropy = np.array(ent_payload["entropy"], dtype=float)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    models = sorted(set().union(*[set(s.keys()) for s in per_seed]))
    ordering_placeholders = {m: [] for m in models}
    strip = not args.keep_provider_prefix
    for model, label, color in iter_series_for_paper_plot(ordering_placeholders, strip_provider=strip):
        arrays = []
        for seed_series in per_seed:
            if model in seed_series:
                arrays.append(to_array(seed_series[model]))
        if not arrays:
            continue
        values = np.vstack(arrays)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)

        ax1.plot(rates_arr, mean, color=color, linewidth=2, marker="o", markersize=2,
                 markerfacecolor=color, markeredgewidth=0, label=label)
        ax1.fill_between(rates_arr, mean - std, mean + std, color=color, alpha=0.15)

    ax2 = ax1.twinx()
    ax2.plot(
        rates_arr, entropy, color="gray", linewidth=1.5, linestyle="--",
        marker="o", markersize=2, markerfacecolor="gray", markeredgewidth=0,
        label="Tokenization Entropy"
    )
    ax2.set_ylabel("Tokenization Entropy (bits)", fontsize=11, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(4, 10)

    style_axes(ax1, title=args.title, xlabel="Word Fragmentation Rate", ylabel="Micro-F1")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, framealpha=0.9, loc="lower left")
    save(fig, args.output_prefix)


if __name__ == "__main__":
    main()
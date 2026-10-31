import json
from typing import Dict, Iterator, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_COLORS = ["#E24B4A", "#2166AC", "#F5A623", "#8B5CF6", "#10B981", "#F97316"]

# Paper-facing legend strings (match main-text Figure 2). Keys are stripped slugs lowercased.
_SLUG_TO_PAPER_DISPLAY = {
    "gemini-3-flash-preview": "Gemini 3.0 Flash",
    "gemini-3.0-flash": "Gemini 3.0 Flash",
    "gemini-3-flash": "Gemini 3.0 Flash",
    "gpt-5.2": "GPT-5.2",
    "claude-haiku-4-5": "Claude 4.5 Haiku",
    "claude-4.5-haiku": "Claude 4.5 Haiku",
}

# Plot order and DEFAULT_COLORS indices 0..2: red, blue, orange (matches Figure 2).
PAPER_DISPLAY_ORDER = ["Gemini 3.0 Flash", "GPT-5.2", "Claude 4.5 Haiku"]


def load_summary(path: str) -> Tuple[List[float], Dict[str, List[float]]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["rates"], payload["series"]


def maybe_strip_provider(name: str, strip: bool = True) -> str:
    if strip and ":" in name:
        return name.split(":", 1)[1]
    return name


def paper_display_core(model_key: str) -> str:
    """
    Canonical paper-facing name from provider:slug keys (strip provider before lookup).
    Unknown models keep the stripped API slug (original casing).
    """
    stripped = maybe_strip_provider(model_key, strip=True)
    slug_l = stripped.strip().lower()
    return _SLUG_TO_PAPER_DISPLAY.get(slug_l, stripped)


def paper_legend_label(model_key: str, strip_provider: bool = True) -> str:
    """Legend text; strip_provider=False prefixes provider but colors still follow paper_display_core()."""
    core = paper_display_core(model_key)
    if strip_provider:
        return core
    if ":" in model_key:
        prefix = model_key.split(":", 1)[0]
        return f"{prefix}:{core}"
    return core


def iter_series_for_paper_plot(
    series: Dict[str, List[float]],
    *,
    strip_provider: bool = True,
) -> Iterator[Tuple[str, str, str]]:
    """
    Yield (model_key, legend_label, color_hex) in stable paper order:
    Gemini, GPT, Claude first (colors 0–2), then remaining models.
    """
    items = list(series.items())

    def sort_key(kv: Tuple[str, List[float]]) -> Tuple[int, int, str, str]:
        key, _vals = kv
        core = paper_display_core(key)
        if core in PAPER_DISPLAY_ORDER:
            return (0, PAPER_DISPLAY_ORDER.index(core), core.lower(), key)
        return (1, 0, core.lower(), key)

    items.sort(key=sort_key)

    unknown_rank = 0
    for model_key, _vals in items:
        core = paper_display_core(model_key)
        label = paper_legend_label(model_key, strip_provider=strip_provider)
        if core in PAPER_DISPLAY_ORDER:
            cidx = PAPER_DISPLAY_ORDER.index(core)
        else:
            cidx = 3 + unknown_rank
            unknown_rank += 1
        color = DEFAULT_COLORS[cidx % len(DEFAULT_COLORS)]
        yield model_key, label, color


def to_array(values: List[float]) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def save(fig, output_prefix: str):
    fig.tight_layout()
    fig.savefig(f"{output_prefix}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_prefix}.pdf/.png")


def style_axes(ax, title: str, xlabel: str, ylabel: str, ylim=(0.1, 1.0)):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim is not None:
        ax.set_ylim(*ylim)

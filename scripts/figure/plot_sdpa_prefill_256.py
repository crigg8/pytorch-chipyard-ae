from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MPLCONFIGDIR = ROOT_DIR / ".matplotlib"
XDG_CACHE_HOME = ROOT_DIR / ".cache"
MPLCONFIGDIR.mkdir(exist_ok=True)
XDG_CACHE_HOME.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

from plot_style import legend_box_kwargs, style_legend_frame


CSV_PATH = ROOT_DIR / ".csv" / "spda_prefill_256.csv"
FIGURE_DIR = ROOT_DIR / "figures"
OUT_PATH = FIGURE_DIR / "Fig6b.pdf"

WIDTH_CM = 4.15
HEIGHT_CM = 4.15
FONT_SIZE_PT = 6.4

SIMPLE_STAGE2 = os.environ.get(
    "PYTORCH_CHIPYARD_SIMPLE_STAGE2", ""
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MODEL_ORDER = ["opt"] if SIMPLE_STAGE2 else ["opt", "pythia", "gpt2", "gpt-neo"]
MODEL_LABELS = {
    "opt": "OPT-125M",
    "pythia": "Pythia-160M",
    "gpt2": "GPT2-124M",
    "gpt-neo": "GPT-Neo-125M",
}

GROUP_STEP = 0.64
LABEL_ROTATION_DEG = 25
LABEL_X_SHIFT_PT = 7.0

SERIES = [
    (2, "Gemmini 2-core", "#B279A2"),
    (4, "Gemmini 4-core", "#D4A6C8"),
]


def cm_to_inch(value: float) -> float:
    return value / 2.54


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "axes.formatter.use_mathtext": True,
            "font.size": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    df = pd.read_csv(CSV_PATH)
    df["cycles_b"] = pd.to_numeric(df["total_kernel_cycles_avg"], errors="coerce") / 1e9

    selected = df[df["variant"] == "fp32"].copy()
    raw_values = {
        cores: selected[selected["cores"] == cores].set_index("model")["cycles_b"]
        for cores, _label, _color in SERIES
    }
    model_order = [
        model
        for model in MODEL_ORDER
        if any(
            model in raw_values[cores].index
            and pd.notna(raw_values[cores].loc[model])
            for cores, _label, _color in SERIES
        )
    ]
    active_series = [
        series
        for series in SERIES
        if any(
            model in raw_values[series[0]].index
            and pd.notna(raw_values[series[0]].loc[model])
            for model in model_order
        )
    ]
    if not model_order or not active_series:
        print(f"[plot][SKIP] {OUT_PATH}: no SDPA cycle measurements")
        return
    values = {
        cores: raw_values[cores].reindex(model_order)
        for cores, _label, _color in active_series
    }

    x = np.arange(len(model_order)) * GROUP_STEP
    width = 0.14

    fig, ax = plt.subplots(
        figsize=(cm_to_inch(WIDTH_CM), cm_to_inch(HEIGHT_CM)),
        dpi=220,
    )
    fig.subplots_adjust(left=0.23, right=0.99, top=0.66, bottom=0.30)

    for idx, (cores, label, color) in enumerate(active_series):
        offset = (idx - (len(active_series) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values[cores],
            width=width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS[model] for model in model_order],
        rotation=LABEL_ROTATION_DEG,
        ha="right",
        rotation_mode="anchor",
    )
    label_offset = mtransforms.ScaledTranslation(
        LABEL_X_SHIFT_PT / 72.0,
        0.0,
        fig.dpi_scale_trans,
    )
    for label in ax.get_xticklabels():
        label.set_transform(label.get_transform() + label_offset)
    ax.tick_params(axis="x", pad=1.0)
    ax.set_ylabel("Cycle(B)", labelpad=0.5, color="white")
    ax.set_xlim(x[0] - 0.33, x[-1] + 0.33)
    ymax = max(float(series.max(skipna=True)) for series in values.values())
    ax.set_ylim(top=ymax * 1.33)
    ax.grid(axis="y", color="#b8b8b8", linestyle="--", linewidth=0.45, alpha=0.85, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("black")
    legend = ax.legend(
        ncols=1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.52),
        **legend_box_kwargs("two", columnspacing=0.0),
    )
    style_legend_frame(legend)
    
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved: {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()

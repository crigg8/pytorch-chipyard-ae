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
from matplotlib.ticker import FuncFormatter

from plot_style import legend_box_kwargs, style_legend_frame


CSV_PATH = ROOT_DIR / ".csv" / "cnn_result.csv"
FIGURE_DIR = ROOT_DIR / "figures"
OUT_PDF = FIGURE_DIR / "Fig6a.pdf"

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
MODEL_ORDER = (
    ["SqueezeNet"]
    if SIMPLE_STAGE2
    else ["ResNet", "AlexNet", "MobileNet", "SqueezeNet"]
)
MODEL_LABELS = {
    "ResNet": "ResNet50",
    "AlexNet": "AlexNet",
    "MobileNet": "MobileNetV2",
    "SqueezeNet": "SqueezeNet",
}

GROUP_STEP = 0.64
LABEL_ROTATION_DEG = 25
LABEL_X_SHIFT_PT = 7.0

SERIES = [
    ("Rocket", "4-core", "Rocket", "#4C78A8"),
    ("Saturn", "2 Saturn", "Saturn", "#72B7B2"),
    ("Gemmini", "Dual Gemmini", "Gemmini", "#B279A2"),
]


def cm_to_inch(value: float) -> float:
    return value / 2.54


def cycle_tick(value: float, _pos: int) -> str:
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}".rstrip("0").rstrip(".")


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
            "legend.title_fontsize": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    df = pd.read_csv(CSV_PATH).replace({"x": np.nan, "": np.nan})
    df = df[df["variant"] == "baseline"].copy()

    values: dict[tuple[str, str], float] = {}
    for _, row in df.iterrows():
        device = row["device"]
        model = row["model"]
        for series_device, column, _label, _color in SERIES:
            if device != series_device:
                continue
            raw_value = pd.to_numeric(row[column], errors="coerce")
            if pd.notna(raw_value):
                values[(model, column)] = float(raw_value) / 1e9

    model_order = [
        model
        for model in MODEL_ORDER
        if any((model, column) in values for _device, column, _label, _color in SERIES)
    ]
    active_series = [
        series
        for series in SERIES
        if any((model, series[1]) in values for model in model_order)
    ]
    if not model_order or not active_series:
        print(f"[plot][SKIP] {OUT_PDF}: no absolute-cycle measurements")
        return

    FIGURE_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(cm_to_inch(WIDTH_CM), cm_to_inch(HEIGHT_CM)),
        dpi=260,
    )
    fig.subplots_adjust(left=0.23, right=0.99, top=0.66, bottom=0.30)

    x = np.arange(len(model_order)) * GROUP_STEP
    width = 0.14

    for idx, (_device, column, label, color) in enumerate(active_series):
        offset = (idx - (len(active_series) - 1) / 2) * width
        y = [values.get((model, column), np.nan) for model in model_order]
        ax.bar(
            x + offset,
            y,
            width=width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_ylim(0.65, 60.0)
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.yaxis.set_major_formatter(FuncFormatter(cycle_tick))
    ax.set_ylabel("Cycle(B)", labelpad=0.5)
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
    ax.set_xlim(x[0] - 0.33, x[-1] + 0.33)
    ax.grid(axis="y", which="major", color="#BDBDBD", linestyle="--", linewidth=0.45, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("black")

    handles, labels = ax.get_legend_handles_labels()
    preferred_order = ["Rocket", "Gemmini", "Saturn"]
    legend_order = [
        labels.index(label) for label in preferred_order if label in labels
    ]
    legend = ax.legend(
        [handles[idx] for idx in legend_order],
        [labels[idx] for idx in legend_order],
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.52),
        **legend_box_kwargs("two"),
    )
    style_legend_frame(legend)

    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()

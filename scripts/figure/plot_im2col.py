from pathlib import Path
import os

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
from matplotlib.ticker import FormatStrFormatter

from plot_style import legend_box_kwargs, style_legend_frame


CSV_PATH = ROOT_DIR / ".csv" / "cnn_result.csv"
FIGURE_DIR = ROOT_DIR / "figures"
OUT_PATH = FIGURE_DIR / "Fig10a.pdf"

WIDTH_CM = 3.0
HEIGHT_CM = 4.15
FONT_SIZE_PT = 6.4

MODEL_ORDER = ["ResNet", "AlexNet", "MobileNet", "SqueezeNet"]
MODEL_LABELS = {
    "ResNet": "ResNet50",
    "AlexNet": "AlexNet",
    "MobileNet": "MobileNetV2",
    "SqueezeNet": "SqueezeNet",
}
GROUP_STEP = 0.54
LABEL_ROTATION_DEG = 25
LABEL_X_SHIFT_PT = 7.0


def cm_to_inch(value: float) -> float:
    return value / 2.54


def load_speedups() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH).replace({"x": np.nan, "": np.nan})
    df["Quad Gemmini"] = pd.to_numeric(df["Quad Gemmini"], errors="coerce")

    rows = []
    for model in MODEL_ORDER:
        sub = df[(df["device"] == "Gemmini") & (df["model"] == model)]
        baseline = sub[sub["variant"] == "baseline"]
        im2col = sub[sub["variant"] == "im2col"]
        if baseline.empty or im2col.empty:
            continue

        baseline_cycles = baseline.iloc[0]["Quad Gemmini"]
        im2col_cycles = im2col.iloc[0]["Quad Gemmini"]
        if pd.isna(baseline_cycles) or pd.isna(im2col_cycles):
            continue
        if float(baseline_cycles) <= 0 or float(im2col_cycles) <= 0:
            continue

        rows.append(
            {
                "model": model,
                "label": MODEL_LABELS[model],
                "direct": 1.0,
                "im2col": float(baseline_cycles) / float(im2col_cycles),
            }
        )

    return pd.DataFrame(rows)


def boxed_legend(ax: plt.Axes) -> None:
    legend = ax.legend(
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        **legend_box_kwargs("one"),
    )
    style_legend_frame(legend)


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

    plot_df = load_speedups()
    if plot_df.empty:
        print(f"[plot][SKIP] {OUT_PATH}: no direct/im2col pairs with a baseline")
        return
    x = np.arange(len(plot_df)) * GROUP_STEP
    width = 0.14

    fig, ax = plt.subplots(
        figsize=(cm_to_inch(WIDTH_CM), cm_to_inch(HEIGHT_CM)),
        dpi=220,
    )
    fig.subplots_adjust(left=0.32, right=0.99, top=0.66, bottom=0.30)

    ax.bar(
        x - width / 2,
        plot_df["direct"],
        width,
        color="#6B6B6B",
        edgecolor="black",
        linewidth=0.45,
        label="Direct",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        plot_df["im2col"],
        width,
        color="#F58518",
        edgecolor="black",
        linewidth=0.45,
        hatch="//",
        label="im2col",
        zorder=3,
    )

    ax.set_ylabel("Norm. Perf.", labelpad=0.5)
    ax.set_ylim(0.0, 1.7)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_df["label"],
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
    ax.set_xlim(x[0] - 0.32, x[-1] + 0.32)
    ax.grid(axis="y", color="#BDBDBD", linestyle="--", linewidth=0.45, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("black")
    boxed_legend(ax)

    FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved: {OUT_PATH}")

    plt.close(fig)


if __name__ == "__main__":
    main()

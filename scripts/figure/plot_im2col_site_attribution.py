from __future__ import annotations

import csv
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
import numpy as np


FIGURE_DIR = ROOT_DIR / "figures"
CSV_PATH = ROOT_DIR / ".csv" / "im2col_site_attribution.csv"
OUT_PATHS = {
    "ResNet50": FIGURE_DIR / "Fig10b.pdf",
    "SqueezeNet": FIGURE_DIR / "Fig10c.pdf",
}
WIDTH_CM = 3.15
HEIGHT_CM = 3.4
FONT_SIZE_PT = 6.4
XLIMS = {
    "ResNet50": (-1300, 260),
    "SqueezeNet": (-125, 225),
}
XTICKS = {
    "ResNet50": [-1200, -600, 0],
    "SqueezeNet": [-100, 0, 100, 200],
}
LABELS = {
    "ResNet50": {
        "7x7 conv": "7x7",
        "1x1 conv": "1x1",
        "3x3 conv": "3x3",
        "prepack": "prepack",
        "others": "other",
        "total": "total",
    },
    "SqueezeNet": {
        "3x3 s2p0": "3x3 s2",
        "Fire 3x3": "Fire 3x3",
        "1x1 conv": "1x1",
        "prepack": "prepack",
        "others": "other",
        "total": "total",
    },
}


def cm_to_inch(value: float) -> float:
    return value / 2.54


def load_data() -> dict[str, list[tuple[str, float, float, float, str]]]:
    data: dict[str, list[tuple[str, float, float, float, str]]] = {}
    with CSV_PATH.open() as csv_file:
        for row in csv.DictReader(csv_file):
            data.setdefault(row["model"], []).append(
                (
                    row["label"],
                    float(row["direct_mcycles"]),
                    float(row["im2col_mcycles"]),
                    float(row["delta_mcycles"]),
                    row["kind"],
                )
            )
    return data


def plot_model(model: str, rows: list[tuple[str, float, float, float, str]]) -> None:
    fig, ax = plt.subplots(
        figsize=(cm_to_inch(WIDTH_CM), cm_to_inch(HEIGHT_CM)),
        dpi=220,
    )
    fig.subplots_adjust(left=0.38, right=0.99, top=0.88, bottom=0.14)

    faster_color = "#2F7E6D"
    slower_color = "#C65A2E"
    aux_color = "#4C78A8"
    total_faster_color = "#1F4E79"
    total_slower_color = "#8F2D1F"

    labels = [LABELS[model][row[0]] for row in rows]
    values = [row[3] for row in rows]
    kinds = [row[4] for row in rows]
    y = np.arange(len(rows))

    colors = []
    for value, kind in zip(values, kinds):
        if kind == "total":
            colors.append(total_faster_color if value < 0 else total_slower_color)
        elif kind in {"aux", "other"}:
            colors.append(aux_color if value < 0 else slower_color)
        else:
            colors.append(faster_color if value < 0 else slower_color)

    bars = ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.45, height=0.56, zorder=3)
    for bar, kind in zip(bars, kinds):
        if kind == "total":
            bar.set_hatch("//")
        elif kind in {"aux", "other"}:
            bar.set_hatch("..")

    ax.axvline(0, color="#D43F35", linestyle="--", linewidth=0.6, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.tick_params(axis="y", pad=1.0)
    ax.tick_params(axis="x", pad=1.0)
    ax.margins(y=0.08)
    ax.set_xlim(*XLIMS[model])
    ax.set_xticks(XTICKS[model])
    ax.grid(axis="x", color="#BDBDBD", linestyle="--", linewidth=0.45, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("black")

    FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_PATHS[model], bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved: {OUT_PATHS[model]}")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "axes.formatter.use_mathtext": True,
            "font.size": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "legend.fontsize": 5.5,
            "figure.titlesize": FONT_SIZE_PT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    data = load_data()
    if not data:
        print("[plot][SKIP] im2col site attribution: no direct/im2col pairs with a baseline")
        return
    for model, rows in data.items():
        plot_model(model, rows)


if __name__ == "__main__":
    main()

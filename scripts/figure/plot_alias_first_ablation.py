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
import pandas as pd

from plot_style import legend_box_kwargs, size_cm, style_legend_frame


CSV_PATH = ROOT_DIR / ".csv" / "alias_first_ablation.csv"
FIGURE_DIR = ROOT_DIR / "figures"
OUT_STEM = "Fig9"

WIDTH_CM = 8.6
HEIGHT_CM = 2.8
FONT_SIZE_PT = 6.4

MODEL_ORDER = [
    "resnet50",
    "alexnet",
    "mobilenetv2",
    "squeezenet",
    "opt",
    "pythia",
    "gpt2",
    "gpt-neo",
]
MODEL_LABELS = {
    "resnet50": "ResNet50",
    "alexnet": "AlexNet",
    "mobilenetv2": "MobileNetV2",
    "squeezenet": "SqueezeNet",
    "opt": "OPT-125M",
    "pythia": "Pythia-160M",
    "gpt2": "GPT-2-124M",
    "gpt-neo": "GPT-Neo-125M",
}


def load_results() -> pd.DataFrame:
    frame = pd.read_csv(CSV_PATH)
    frame["speedup"] = pd.to_numeric(frame["speedup"], errors="coerce")
    frame = frame.dropna(subset=["speedup"]).copy()
    frame = frame[frame["speedup"] > 0]
    frame["order"] = frame["model"].map(
        {model: index for index, model in enumerate(MODEL_ORDER)}
    )
    frame = frame.dropna(subset=["order"]).sort_values("order")
    frame["label"] = frame["model"].map(MODEL_LABELS)

    return frame[["model", "label", "speedup", "order"]]


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

    frame = load_results()
    if frame.empty:
        print(f"[plot][SKIP] {FIGURE_DIR / (OUT_STEM + '.pdf')}: no complete ON/OFF pairs")
        return

    is_llm = frame["model"].isin(MODEL_ORDER[4:]).to_numpy()
    x = np.arange(len(frame), dtype=float)
    transition = next((index for index, value in enumerate(is_llm) if value), None)
    if transition is not None and transition > 0:
        x[transition:] += 0.35
    width = 0.26

    fig, ax = plt.subplots(figsize=size_cm(WIDTH_CM, HEIGHT_CM), dpi=300)
    fig.subplots_adjust(left=0.145, right=0.99, top=0.88, bottom=0.40)

    off_bars = ax.bar(
        x - width / 2,
        np.ones(len(frame)),
        width,
        color="#7B3294",
        edgecolor="black",
        linewidth=0.45,
        label="Alias-first off",
        zorder=3,
    )
    bars = ax.bar(
        x + width / 2,
        frame["speedup"],
        width,
        color="#F2C12E",
        edgecolor="black",
        linewidth=0.45,
        label="Alias-first on",
        zorder=3,
    )

    upper = max(1.5, float(frame["speedup"].max()) * 1.18)
    ax.set_ylim(0.0, upper)
    ax.set_yticks([0.0, 1.0, 2.0, 3.0])
    ax.set_ylabel("Norm. Perf.", labelpad=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        frame["label"], rotation=25, ha="right", rotation_mode="anchor"
    )
    ax.tick_params(axis="x", pad=1.0)
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    ax.grid(
        axis="y",
        color="#B8B8B8",
        linestyle="--",
        linewidth=0.45,
        alpha=0.85,
        zorder=0,
    )
    if transition is not None and transition > 0:
        ax.axvline(
            (x[transition - 1] + x[transition]) / 2,
            color="#8A8A8A",
            linewidth=0.45,
            zorder=1,
        )
    ax.set_axisbelow(True)

    for bar, value in zip(bars, frame["speedup"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + upper * 0.018,
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=5.5,
            fontweight="bold",
        )

    legend = fig.legend(
        handles=(off_bars[0], bars[0]),
        labels=("Alias-first off", "Alias-first on"),
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.57, 0.985),
        borderaxespad=0.0,
        **legend_box_kwargs("one", fontsize=5.5),
    )
    style_legend_frame(legend)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    FIGURE_DIR.mkdir(exist_ok=True)
    output = FIGURE_DIR / f"{OUT_STEM}.pdf"
    fig.savefig(output, dpi=300, facecolor="white")
    print(f"Saved: {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import os
import re

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
MPLCONFIGDIR = ROOT_DIR / ".matplotlib"
XDG_CACHE_HOME = ROOT_DIR / ".cache"
MPLCONFIGDIR.mkdir(exist_ok=True)
XDG_CACHE_HOME.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_style import apply_plot_style, legend_box_kwargs, size_cm, style_legend_frame

apply_plot_style()


FIGURE_DIR = ROOT_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)
WORKSPACE_DIR = ROOT_DIR.parent
LOG_DIR = Path(os.environ.get("PYTORCH_CHIPYARD_LOG_DIR", WORKSPACE_DIR / "examples" / ".logs"))

LOG_GLOB = "*gemmini-max-autotune*4core-autotune.log"
OUT_STEM = "Fig11"

SIMPLE_STAGE2 = os.environ.get(
    "PYTORCH_CHIPYARD_SIMPLE_STAGE2", ""
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BASELINE_BLOCKING = (64, 64, 512) if SIMPLE_STAGE2 else (64, 64, 32)
BLACK = "#111111"
BASELINE_RED = "#D43F35"
DEFAULT_RED = "#D43F35"
GRID_GRAY = "#BDBDBD"
BEST_GRAY = "#6F6F6F"
LEGEND_FONT_SIZE_PT = 9.6


def _extract(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _load_log_candidates(path: Path) -> pd.DataFrame:
    records: list[dict[str, int]] = []
    text = path.read_text()
    for block in text.split("\nAutotune Candidate\n")[1:]:
        candidate = _extract(r"^Candidate: (\d+)$", block)
        bm = _extract(r"^  BLOCK_M=(\d+)$", block)
        bn = _extract(r"^  BLOCK_N=(\d+)$", block)
        bk = _extract(r"^  BLOCK_K=(\d+)$", block)
        cycles = _extract(r"^Cycles: (\d+)$", block)
        tile_i = _extract(r"^  gemmini_tile_i=(\d+)$", block)
        tile_j = _extract(r"^  gemmini_tile_j=(\d+)$", block)
        tile_k = _extract(r"^  gemmini_tile_k=(\d+)$", block)
        if not (candidate and bm and bn and bk and cycles and tile_i and tile_j and tile_k):
            continue
        records.append(
            {
                "candidate": int(candidate),
                "bm": int(bm),
                "bn": int(bn),
                "bk": int(bk),
                "cycles": int(cycles),
                "tile_i": int(tile_i),
                "tile_j": int(tile_j),
                "tile_k": int(tile_k),
            }
        )

    if not records:
        raise RuntimeError(f"No autotune candidates parsed from {path}")

    df = pd.DataFrame.from_records(records)
    if SIMPLE_STAGE2:
        df = df[
            (df["bm"] == BASELINE_BLOCKING[0])
            & (df["bn"] == BASELINE_BLOCKING[1])
            & (df["bk"] == BASELINE_BLOCKING[2])
        ].copy()
        if df.empty:
            raise RuntimeError(f"Blocking {BASELINE_BLOCKING} not found in {path}")
    baseline_df = df[
        (df["bm"] == BASELINE_BLOCKING[0])
        & (df["bn"] == BASELINE_BLOCKING[1])
        & (df["bk"] == BASELINE_BLOCKING[2])
        & (df["tile_i"] == 0)
        & (df["tile_j"] == 0)
        & (df["tile_k"] == 0)
    ]
    if baseline_df.empty:
        raise RuntimeError(f"Baseline blocking {BASELINE_BLOCKING} not found in {path}")

    baseline_cycles = int(baseline_df.iloc[0]["cycles"])
    df["speedup"] = baseline_cycles / df["cycles"]
    df["source_log"] = path.name
    df["baseline_cycles"] = baseline_cycles
    return df


def _load_candidates() -> pd.DataFrame:
    log_paths = sorted(LOG_DIR.glob(LOG_GLOB))
    if not log_paths:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in log_paths:
        try:
            frames.append(_load_log_candidates(path))
        except (OSError, RuntimeError) as exc:
            print(f"[plot][warn] skipping {path}: {exc}")
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    dedup_keys = ["bm", "bn", "bk", "tile_i", "tile_j", "tile_k"]
    best_idx = df.groupby(dedup_keys)["speedup"].idxmax()
    df = df.loc[best_idx].copy()
    df["blocking"] = list(zip(df["bm"], df["bn"], df["bk"]))
    df.sort_values(["bm", "bn", "bk", "tile_i", "tile_j", "tile_k"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _blocking_order(df: pd.DataFrame) -> list[tuple[int, int, int]]:
    best_by_blocking = (
        df.groupby(["bm", "bn", "bk"], as_index=False)
        .agg(best_speedup=("speedup", "max"), candidate_count=("speedup", "size"))
        .sort_values(["best_speedup", "bm", "bn", "bk"], ascending=[False, True, True, True])
    )
    return [
        (int(row.bm), int(row.bn), int(row.bk))
        for row in best_by_blocking.itertuples(index=False)
    ]


def _format_blocking(blocking: tuple[int, int, int]) -> str:
    bm, bn, bk = blocking
    return rf"({bm},{bn},{bk})"


def _plot(df: pd.DataFrame) -> None:
    order = _blocking_order(df)
    y_index = {blocking: idx for idx, blocking in enumerate(order)}
    rng = np.random.default_rng(17)

    plot_df = df.copy()
    plot_df["y"] = plot_df["blocking"].map(y_index)
    plot_df["is_default"] = (
        (plot_df["tile_i"] == 0)
        & (plot_df["tile_j"] == 0)
        & (plot_df["tile_k"] == 0)
    )
    group_sizes = plot_df.groupby("blocking")["speedup"].transform("size")
    jitter = rng.uniform(-0.24, 0.24, len(plot_df))
    plot_df["y_jitter"] = plot_df["y"] + np.where(
        (group_sizes > 1) & ~plot_df["is_default"],
        jitter,
        0.0,
    )
    explicit_df = plot_df[~plot_df["is_default"]]
    default_df = plot_df[plot_df["is_default"]]

    best_df = (
        plot_df.sort_values("cycles")
        .groupby("blocking", as_index=False)
        .first()
        .sort_values("y")
    )

    height_cm = 0.42 * len(order) + 3.05
    fig, ax = plt.subplots(figsize=size_cm(15.2, height_cm), dpi=300)
    fig.subplots_adjust(left=0.205, right=0.995, top=0.865, bottom=0.125)

    ax.axvline(1.0, color=BASELINE_RED, linestyle="--", linewidth=1.25, zorder=1)
    ax.scatter(
        explicit_df["speedup"],
        explicit_df["y_jitter"],
        s=8.0,
        c=BLACK,
        alpha=0.34,
        linewidths=0,
        zorder=3,
    )
    ax.scatter(
        default_df["speedup"],
        default_df["y"],
        marker="s",
        s=22.0,
        c=DEFAULT_RED,
        alpha=0.95,
        edgecolors="white",
        linewidths=0.35,
        zorder=5,
    )
    for _, row in best_df.iterrows():
        ax.plot(
            [row["speedup"], row["speedup"]],
            [row["y"] - 0.22, row["y"] + 0.22],
            color=BEST_GRAY,
            linewidth=0.65,
            zorder=2,
        )

    ax.set_xlim(0.0, 5.1)
    ax.set_xticks(np.arange(0, 5.1, 1.0))
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_format_blocking(blocking) for blocking in order], fontsize=8.4)
    ax.set_xlabel("Normalized performance", fontsize=11.5, labelpad=4)
    ax.set_ylabel(r"Blocking $(B_M,B_N,B_K)$", fontsize=11.5, labelpad=4)
    ax.grid(axis="x", color=GRID_GRAY, linestyle="--", linewidth=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9.0, width=1.1, length=5)
    ax.tick_params(axis="y", width=1.1, length=4, pad=2)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=BLACK,
            markeredgewidth=0,
            alpha=0.42,
            markersize=6.8,
            label="Explicit tile",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor=DEFAULT_RED,
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=7.4,
            label="Buddy default",
        ),
        Line2D(
            [0],
            [0],
            color=BASELINE_RED,
            linestyle="--",
            linewidth=1.8,
            label="Baseline",
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        ncols=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.035),
        borderaxespad=0.0,
        **legend_box_kwargs(
            "one",
            columnspacing=0.55,
            handlelength=0.70,
            handletextpad=0.18,
            fontsize=LEGEND_FONT_SIZE_PT,
        ),
    )
    style_legend_frame(legend)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)

    out_pdf = FIGURE_DIR / f"{OUT_STEM}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    print(f"Saved: {out_pdf}")


def main() -> None:
    df = _load_candidates()
    if df.empty:
        print(
            f"[plot][SKIP] {FIGURE_DIR / (OUT_STEM + '.pdf')}: "
            f"no candidates with baseline blocking {BASELINE_BLOCKING}"
        )
        return
    print(
        "Parsed "
        f"{len(df)} unique candidates across "
        f"{df[['bm', 'bn', 'bk']].drop_duplicates().shape[0]} blockings"
    )
    print(
        "Best speedup: "
        f"{df['speedup'].max():.2f}x; slowest point: {df['speedup'].min():.2f}x"
    )
    _plot(df)


if __name__ == "__main__":
    main()

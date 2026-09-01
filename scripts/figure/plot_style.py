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

ONE_COL_WIDTH_CM = 18.0
TWO_COL_WIDTH_CM = 18.0
LEGEND_FONT_SIZE_PT = 5.5
LEGEND_FRAME_LINEWIDTH = 0.6

_LEGEND_COMMON = {
    "frameon": True,
    "fancybox": False,
    "framealpha": 1.0,
    "edgecolor": "black",
    "fontsize": LEGEND_FONT_SIZE_PT,
}

_ONE_ROW_LEGEND = {
    "columnspacing": 0.25,
    "handlelength": 0.70,
    "handletextpad": 0.18,
    "borderpad": 0.16,
    "labelspacing": 0.18,
}

_TWO_ROW_LEGEND = {
    "columnspacing": 0.25,
    "handlelength": 0.85,
    "handletextpad": 0.25,
    "borderpad": 0.22,
    "labelspacing": 0.18,
}


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "axes.formatter.use_mathtext": True,
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": LEGEND_FONT_SIZE_PT,
            "legend.title_fontsize": LEGEND_FONT_SIZE_PT,
            "figure.titlesize": 18,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def legend_box_kwargs(rows: str, **overrides) -> dict:
    if rows not in {"one", "two"}:
        raise ValueError("rows must be 'one' or 'two'")
    kwargs = dict(_LEGEND_COMMON)
    kwargs.update(_ONE_ROW_LEGEND if rows == "one" else _TWO_ROW_LEGEND)
    kwargs.update(overrides)
    return kwargs


def style_legend_frame(legend) -> None:
    legend.get_frame().set_linewidth(LEGEND_FRAME_LINEWIDTH)


def size_from_width_cm(width_cm: float, original_width_in: float, original_height_in: float) -> tuple[float, float]:
    width_in = width_cm / 2.54
    height_in = width_in * (original_height_in / original_width_in)
    return (width_in, height_in)


def cm_to_inch(value: float) -> float:
    return value / 2.54


def size_cm(width_cm: float = 18.0, height_cm: float = 6.0) -> tuple[float, float]:
    return (cm_to_inch(width_cm), cm_to_inch(height_cm))

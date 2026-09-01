from __future__ import annotations

import os
import re
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = ROOT_DIR.parent
LOG_DIR = Path(os.environ.get("PYTORCH_CHIPYARD_LOG_DIR", WORKSPACE_DIR / "examples" / ".logs"))
FIGURE_DIR = ROOT_DIR / "figures"
ARTIFACT_ROOT = WORKSPACE_DIR / "examples"
LEGACY_IR_DIR = WORKSPACE_DIR / "triton_chipyard" / "IR"

MPLCONFIGDIR = ROOT_DIR / ".matplotlib"
XDG_CACHE_HOME = ROOT_DIR / ".cache"
MPLCONFIGDIR.mkdir(exist_ok=True)
XDG_CACHE_HOME.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_style import legend_box_kwargs, style_legend_frame


SUBFIG_WIDTH = 0.50
BASE_SUBFIG_WIDTH = 0.44
BASE_PANEL_WIDTH_CM = 4.45
PANEL_WIDTH_CM = BASE_PANEL_WIDTH_CM * SUBFIG_WIDTH / BASE_SUBFIG_WIDTH
PANEL_HEIGHT_CM = 4.30
FONT_SIZE_PT = 6.4

AXES_LEFT = 0.23
AXES_RIGHT = 0.985
AXES_TOP = 0.78
AXES_BOTTOM = 0.20
LEGEND_Y = 1.29

CATEGORIES = [
    "Large conv/mm",
    "Depthwise conv",
    "Point/reduce/other",
    "Small mm",
]

CATEGORY_COLORS = {
    "Large conv/mm": "#4C78A8",
    "Small mm": "#F58518",
    "Depthwise conv": "#E45756",
    "Point/reduce/other": "#8C8C8C",
}

ANNOTATION_RED = "#C00000"

LEGEND_LABELS = {
    "Large conv/mm": "Large MM/Conv",
    "Small mm": "Small MM",
    "Depthwise conv": "Depthwise Conv",
    "Point/reduce/other": "Other",
}


@dataclass(frozen=True)
class RunSpec:
    model: str
    backend: str
    model_log: str
    autotune_log: str


RUNS = [
    RunSpec(
        "MobileNetV2",
        "Rocket 4",
        "mobilenetv2-rocket-4core-model.log",
        "mobilenetv2-rocket-4core-autotune.log",
    ),
    RunSpec(
        "MobileNetV2",
        "Gemmini 2",
        "mobilenetv2-gemmini-2core-model.log",
        "mobilenetv2-gemmini-2core-autotune.log",
    ),
    RunSpec(
        "SqueezeNet",
        "Rocket 4",
        "squeezenet-rocket-4core-model.log",
        "squeezenet-rocket-4core-autotune.log",
    ),
    RunSpec(
        "SqueezeNet",
        "Gemmini 2",
        "squeezenet-gemmini-2core-model.log",
        "squeezenet-gemmini-2core-autotune.log",
    ),
]

SIMPLE_STAGE2 = os.environ.get(
    "PYTORCH_CHIPYARD_SIMPLE_STAGE2", ""
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
if SIMPLE_STAGE2:
    RUNS = [spec for spec in RUNS if spec.model == "SqueezeNet"]

ROCKET_SCALING_RUNS = [
    RunSpec(
        "MobileNetV2",
        "4",
        "mobilenetv2-rocket-4core-model.log",
        "mobilenetv2-rocket-4core-autotune.log",
    ),
    RunSpec(
        "MobileNetV2",
        "8",
        "mobilenetv2-rocket-8core-model.log",
        "mobilenetv2-rocket-8core-autotune.log",
    ),
    RunSpec(
        "MobileNetV2",
        "16",
        "mobilenetv2-rocket-16core-model.log",
        "mobilenetv2-rocket-16core-autotune.log",
    ),
    RunSpec(
        "AlexNet",
        "4",
        "alexnet-rocket-4core-model.log",
        "alexnet-rocket-4core-autotune.log",
    ),
    RunSpec(
        "AlexNet",
        "8",
        "alexnet-rocket-8core-model.log",
        "alexnet-rocket-8core-autotune.log",
    ),
    RunSpec(
        "AlexNet",
        "16",
        "alexnet-rocket-16core-model.log",
        "alexnet-rocket-16core-autotune.log",
    ),
]

def cm_to_inch(value: float) -> float:
    return value / 2.54


def parse_model_log(path: Path) -> tuple[float, int, list[tuple[str, int]]]:
    text = path.read_text(errors="replace")
    avg_match = re.search(r"Avg Model cycle:\s*([0-9.]+)", text)
    samples_match = re.search(r"Model samples:\s*(\d+)", text)
    if avg_match is None or samples_match is None:
        raise ValueError(f"could not parse model summary: {path}")

    stats_text = text.split("All Kernel Cycle Stats (execution order)", 1)[1]
    rows: list[tuple[str, int]] = []
    for entry in re.split(r"\n(?=\d+\. )", stats_text.strip()):
        header = re.match(r"\d+\.\s+(\S+)", entry)
        total = re.search(r"Total launch cycle:\s*(\d+)", entry)
        if header is None or total is None:
            continue
        rows.append((header.group(1), int(total.group(1))))
    return float(avg_match.group(1)), int(samples_match.group(1)), rows


def parse_autotune_metadata(path: Path) -> dict[str, dict[str, str | bool]]:
    text = path.read_text(errors="replace")
    metadata: dict[str, dict[str, str | bool]] = {}
    for block in text.split("Autotune Candidate\n")[1:]:
        kernel = re.search(r"Kernel:\s*(\S+)", block)
        raw = re.search(r"Raw kernel:\s*(\S+)", block)
        normalized = re.search(r"Normalized kernel:\s*(\S+)", block)
        if kernel is None or raw is None:
            continue
        metadata[kernel.group(1)] = {
            "raw": raw.group(1),
            "normalized": normalized.group(1) if normalized else "",
            "uses_tl_dot": "uses_tl_dot=True" in block,
        }
    return metadata


def first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    formatted = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"model_spec.json not found. Tried:\n  {formatted}")


def artifact_path(model: str, *parts: str) -> Path:
    return ARTIFACT_ROOT / f"artifact-{model}" / Path(*parts)


def model_spec_path(spec: RunSpec) -> Path:
    if spec.model == "AlexNet":
        return first_existing_path(
            [
                artifact_path("alexnet", "scalar", "model_spec.json"),
                artifact_path("alexnet", "rocket", "model_spec.json"),
                ARTIFACT_ROOT / "alexnet" / "scalar" / "model_spec.json",
                ARTIFACT_ROOT / "alexnet" / "rocket" / "model_spec.json",
                LEGACY_IR_DIR / "alexnet-rocket" / "model_spec.json",
            ]
        )
    if spec.model == "MobileNetV2" and spec.backend == "Gemmini 2":
        return first_existing_path(
            [
                artifact_path("mobilenetv2", "gemmini", "model_spec.json"),
                artifact_path("mobilenet", "gemmini", "model_spec.json"),
                ARTIFACT_ROOT / "mobilenetv2" / "gemmini" / "model_spec.json",
                ARTIFACT_ROOT / "mobilenet" / "gemmini" / "model_spec.json",
                LEGACY_IR_DIR / "mobilenet-gemmini" / "model_spec.json",
            ]
        )
    if spec.model == "MobileNetV2":
        return first_existing_path(
            [
                artifact_path("mobilenetv2", "scalar", "model_spec.json"),
                artifact_path("mobilenetv2", "rocket", "model_spec.json"),
                artifact_path("mobilenet", "scalar", "model_spec.json"),
                artifact_path("mobilenet", "rocket", "model_spec.json"),
                ARTIFACT_ROOT / "mobilenetv2" / "scalar" / "model_spec.json",
                ARTIFACT_ROOT / "mobilenetv2" / "rocket" / "model_spec.json",
                ARTIFACT_ROOT / "mobilenet" / "scalar" / "model_spec.json",
                ARTIFACT_ROOT / "mobilenet" / "rocket" / "model_spec.json",
                LEGACY_IR_DIR / "mobilenet-rocket" / "model_spec.json",
            ]
        )
    if spec.model == "SqueezeNet" and spec.backend == "Gemmini 2":
        return first_existing_path(
            [
                artifact_path("squeezenet", "gemmini", "model_spec.json"),
                ARTIFACT_ROOT / "squeezenet" / "gemmini" / "model_spec.json",
                LEGACY_IR_DIR / "squeezenet-gemmini" / "model_spec.json",
            ]
        )
    if spec.model == "SqueezeNet":
        return first_existing_path(
            [
                artifact_path("squeezenet", "scalar", "model_spec.json"),
                artifact_path("squeezenet", "rocket", "model_spec.json"),
                ARTIFACT_ROOT / "squeezenet" / "scalar" / "model_spec.json",
                ARTIFACT_ROOT / "squeezenet" / "rocket" / "model_spec.json",
                LEGACY_IR_DIR / "squeezenet-rocket" / "model_spec.json",
            ]
        )
    raise ValueError(f"no model spec mapping for {spec.model} {spec.backend}")


def load_shape_order(spec_path: Path) -> list[tuple[int, int, int] | None]:
    model_spec = json.loads(spec_path.read_text())
    buffers = {item["name"]: item for item in model_spec["buffers"]}
    launch_steps = [step for step in model_spec["steps"] if step.get("kind") == "launch"]

    def shape_of(name: str) -> list[int] | None:
        buffer = buffers.get(name)
        return buffer.get("size_hint") if buffer else None

    def infer_shape(step: dict) -> tuple[int, int, int] | None:
        raw_name = step.get("triton_meta", {}).get("chipyard_default_kernel_name", "")
        if not raw_name.startswith("triton_mm"):
            return None

        weights: list[tuple[str, list[int]]] = []
        output: tuple[str, list[int]] | None = None
        for arg in step["call_args"]:
            if arg["kind"] == "constant":
                shape = shape_of(arg["name"])
                if shape and len(shape) == 4 and shape[2:] == [1, 1]:
                    weights.append((arg["name"], shape))
                elif shape and len(shape) == 2:
                    weights.append((arg["name"], shape))
            elif arg["kind"] in {"buffer", "output"}:
                shape = shape_of(arg["name"])
                if shape:
                    output = (arg["name"], shape)

        if not weights or output is None:
            return None

        _, output_shape = output
        chosen: tuple[int, int] | None = None
        for _, weight_shape in weights:
            if len(weight_shape) == 4:
                n, k = weight_shape[0], weight_shape[1]
                if (len(output_shape) == 2 and output_shape[-1] == n) or (
                    len(output_shape) == 4 and output_shape[1] == n
                ):
                    chosen = (n, k)
                    break
            elif len(weight_shape) == 2:
                if len(output_shape) >= 2 and output_shape[-1] == weight_shape[1]:
                    chosen = (weight_shape[1], weight_shape[0])
                    break
                if len(output_shape) >= 2 and output_shape[-1] == weight_shape[0]:
                    chosen = (weight_shape[0], weight_shape[1])
                    break

        if chosen is None:
            weight_shape = weights[-1][1]
            chosen = (
                (weight_shape[0], weight_shape[1])
                if len(weight_shape) == 4
                else (weight_shape[1], weight_shape[0])
            )

        n, k = chosen
        if len(output_shape) == 2:
            m = output_shape[0]
        elif len(output_shape) == 4:
            m = output_shape[0] * output_shape[2] * output_shape[3]
        else:
            m = output_shape[0]
        return (m, k, n)

    return [infer_shape(step) for step in launch_steps]


def classify_kernel(
    kernel: str,
    metadata: dict[str, str | bool] | None,
    shape: tuple[int, int, int] | None,
) -> str:
    raw = str(metadata["raw"]) if metadata is not None else ""
    if raw.startswith("triton_depthwise_convolution2d"):
        return "Depthwise conv"
    if raw.startswith("triton_mm"):
        if shape is not None and shape[0] <= 64 and shape[2] <= 320:
            return "Small mm"
        return "Large conv/mm"
    if raw.startswith("triton_convolution2d"):
        return "Large conv/mm"
    if kernel.startswith("triton_tem_"):
        return "Large conv/mm"
    return "Point/reduce/other"


def load_run(spec: RunSpec) -> dict:
    avg, samples, kernel_rows = parse_model_log(LOG_DIR / spec.model_log)
    metadata = parse_autotune_metadata(LOG_DIR / spec.autotune_log)
    shape_order = load_shape_order(model_spec_path(spec))
    totals = defaultdict(float)
    for index, (kernel, total_cycles) in enumerate(kernel_rows):
        shape = shape_order[index] if index < len(shape_order) else None
        category = classify_kernel(kernel, metadata.get(kernel), shape)
        totals[category] += total_cycles / samples / 1e9

    logged_total = sum(totals.values())
    avg_b = avg / 1e9
    if avg_b > logged_total:
        totals["Point/reduce/other"] += avg_b - logged_total

    return {
        "model": spec.model,
        "backend": spec.backend,
        "avg_b": avg_b,
        "totals": {category: totals[category] for category in CATEGORIES},
        "logged_total_b": logged_total,
    }


def load_rocket_scaling_runs() -> list[dict]:
    return [load_run(spec) | {"core": spec.backend} for spec in ROCKET_SCALING_RUNS]


def load_available_run(spec: RunSpec, include_core: bool = False) -> dict | None:
    try:
        run = load_run(spec)
    except (FileNotFoundError, OSError, ValueError, IndexError) as exc:
        print(
            f"[plot][warn] skipping {spec.model} {spec.backend}: {exc}"
        )
        return None
    return run | ({"core": spec.backend} if include_core else {})


def apply_style() -> None:
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


def style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("black")


def add_category_legend(ax: plt.Axes, y: float = LEGEND_Y) -> None:
    legend_order = [
        "Large conv/mm",
        "Depthwise conv",
        "Small mm",
        "Point/reduce/other",
    ]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[category], ec="black", lw=0.45)
        for category in legend_order
    ]
    legend = ax.legend(
        handles,
        [LEGEND_LABELS[category] for category in legend_order],
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        **legend_box_kwargs("two", columnspacing=0.62, labelspacing=0.16),
    )
    style_legend_frame(legend)


def draw_backend_panel(ax: plt.Axes, runs: list[dict]) -> None:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_model[run["model"]].append(run)

    ordered_runs: list[dict] = []
    y_positions: list[float] = []
    tick_labels: list[str] = []
    group_centers: list[float] = []
    group_labels: list[str] = []
    y_cursor = 0.0
    backend_labels = {"Rocket 4": "R4", "Gemmini 2": "G2"}
    for model in ["MobileNetV2", "SqueezeNet"]:
        model_runs = by_model.get(model, [])
        if not model_runs:
            continue
        model_runs.sort(key=lambda run: ["Rocket 4", "Gemmini 2"].index(run["backend"]))
        start = y_cursor
        for run in model_runs:
            ordered_runs.append(run)
            y_positions.append(y_cursor)
            tick_labels.append(backend_labels[run["backend"]])
            y_cursor += 0.30
        group_centers.append((start + y_positions[-1]) / 2)
        group_labels.append(model)
        y_cursor += 0.35

    y = np.array(y_positions)
    height = 0.18
    lefts = np.zeros(len(ordered_runs))
    for category in CATEGORIES:
        values = np.array([run["totals"][category] for run in ordered_runs])
        ax.barh(
            y,
            values,
            height=height,
            left=lefts,
            color=CATEGORY_COLORS[category],
            edgecolor="black",
            linewidth=0.45,
            label=category,
            zorder=3,
        )
        lefts += values

    ax.set_xlabel("Cycle (B)", labelpad=0.5)
    ax.set_xlim(0.0, max(run["avg_b"] for run in ordered_runs) * 1.12)
    ax.set_xticks([0.0, 2.5, 5.0])
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels)
    ax.tick_params(axis="y", pad=1.0)
    ax.tick_params(axis="x", pad=1.0)
    ax.set_ylim(y[-1] + 0.18, y[0] - 0.18)
    ax.grid(axis="x", color="#BDBDBD", linestyle="--", linewidth=0.45, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    for center, label in zip(group_centers, group_labels):
        ax.text(
            -0.18,
            center,
            label,
            ha="center",
            va="center",
            rotation=90,
            transform=ax.get_yaxis_transform(),
            fontsize=5.8,
        )


def draw_scaling_panel(ax: plt.Axes, runs: list[dict]) -> None:
    y_positions: list[float] = []
    tick_labels: list[str] = []
    by_model: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_model[run["model"]].append(run)

    y_cursor = 0.0
    ordered_runs: list[dict] = []
    group_centers: list[float] = []
    group_names: list[str] = []
    for model_label in ["MobileNetV2", "AlexNet"]:
        rows = by_model.get(model_label, [])
        baseline = next(
            (row for row in rows if row["core"] == "4" and row["avg_b"] > 0),
            None,
        )
        if baseline is None:
            continue
        rows.sort(key=lambda row: int(row["core"]))
        group_start = y_cursor
        for index, row in enumerate(rows):
            ordered_runs.append(row)
            y_positions.append(y_cursor)
            tick_labels.append(row["core"])
            y_cursor += 0.26
        group_end = y_cursor - 0.26
        group_centers.append((group_start + group_end) / 2)
        group_names.append(model_label)
        y_cursor += 0.58

    y = np.array(y_positions)
    height = 0.155
    lefts = np.zeros(len(ordered_runs))
    small_mm_segments: list[tuple[float, dict, float, float]] = []
    for category in CATEGORIES:
        values = np.array(
            [
                run["totals"][category]
                / next(
                    row["avg_b"]
                    for row in by_model[run["model"]]
                    if row["core"] == "4"
                )
                for run in ordered_runs
            ]
        )
        if category == "Small mm":
            small_mm_segments = [
                (ypos, run, left, value)
                for ypos, run, left, value in zip(y, ordered_runs, lefts, values)
            ]
        ax.barh(
            y,
            values,
            height=height,
            left=lefts,
            color=CATEGORY_COLORS[category],
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
        lefts += values

    for ypos, run, left, value in small_mm_segments:
        if run["model"] != "MobileNetV2" or run["core"] == "4" or value <= 0.0:
            continue
        baseline = next(
            row for row in by_model[run["model"]] if row["core"] == "4"
        )
        base_small = baseline["totals"]["Small mm"]
        speedup = base_small / run["totals"]["Small mm"]
        segment_start = left
        segment_end = left + value
        bracket_y = ypos - height * 0.72
        tick_y = ypos - height * 0.28
        ax.plot(
            [segment_start, segment_end],
            [bracket_y, bracket_y],
            color=ANNOTATION_RED,
            linewidth=0.75,
            clip_on=False,
            zorder=6,
        )
        ax.plot(
            [segment_start, segment_start],
            [bracket_y, tick_y],
            color=ANNOTATION_RED,
            linewidth=0.75,
            clip_on=False,
            zorder=6,
        )
        ax.plot(
            [segment_end, segment_end],
            [bracket_y, tick_y],
            color=ANNOTATION_RED,
            linewidth=0.75,
            clip_on=False,
            zorder=6,
        )
        ax.text(
            segment_end + 0.018,
            ypos,
            rf"$\leftarrow$ {speedup:.1f}x vs. baseline",
            ha="left",
            va="center",
            fontsize=4.65,
            color=ANNOTATION_RED,
            zorder=6,
        )

    ax.set_xlabel("Norm. cycle", labelpad=0.5)
    ax.set_xlim(0.0, 1.08)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels)
    ax.tick_params(axis="y", pad=1.0)
    ax.tick_params(axis="x", pad=1.0)
    ax.set_ylim(y[-1] + 0.20, y[0] - 0.20)
    ax.grid(axis="x", color="#BDBDBD", linestyle="--", linewidth=0.45, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    for center, label in zip(group_centers, group_names):
        ax.text(
            -0.18,
            center,
            label,
            ha="center",
            va="center",
            rotation=90,
            transform=ax.get_yaxis_transform(),
            fontsize=5.8,
        )


def draw_figure(backend_runs: list[dict], scaling_runs: list[dict]) -> None:
    apply_style()
    FIGURE_DIR.mkdir(exist_ok=True)

    if backend_runs:
        fig, ax = plt.subplots(
            figsize=(cm_to_inch(PANEL_WIDTH_CM), cm_to_inch(PANEL_HEIGHT_CM)),
            dpi=260,
        )
        fig.subplots_adjust(
            left=AXES_LEFT, right=AXES_RIGHT, top=AXES_TOP, bottom=AXES_BOTTOM
        )
        draw_backend_panel(ax, backend_runs)
        style_axes(ax)
        add_category_legend(ax)
        out_path = FIGURE_DIR / "Fig8a.pdf"
        fig.savefig(out_path)
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        print("[plot][SKIP] backend attribution: no usable model runs")

    scaling_models = {
        run["model"]
        for run in scaling_runs
        if run["core"] == "4" and run["avg_b"] > 0
    }
    scaling_runs = [run for run in scaling_runs if run["model"] in scaling_models]
    if scaling_runs:
        fig, ax = plt.subplots(
            figsize=(cm_to_inch(PANEL_WIDTH_CM), cm_to_inch(PANEL_HEIGHT_CM)),
            dpi=260,
        )
        fig.subplots_adjust(
            left=AXES_LEFT, right=AXES_RIGHT, top=AXES_TOP, bottom=AXES_BOTTOM
        )
        draw_scaling_panel(ax, scaling_runs)
        style_axes(ax)
        add_category_legend(ax)
        out_path = FIGURE_DIR / "Fig8b.pdf"
        fig.savefig(out_path)
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        print("[plot][SKIP] scaling attribution: no runs with a 4-core baseline")


def main() -> None:
    backend_runs = [
        run for spec in RUNS if (run := load_available_run(spec)) is not None
    ]
    scaling_runs = [] if SIMPLE_STAGE2 else [
        run
        for spec in ROCKET_SCALING_RUNS
        if (run := load_available_run(spec, include_core=True)) is not None
    ]
    draw_figure(backend_runs, scaling_runs)
    for run in backend_runs:
        pieces = ", ".join(
            f"{category}={run['totals'][category]:.3f}B"
            for category in CATEGORIES
            if run["totals"][category] > 0.001
        )
        print(f"{run['model']} {run['backend']}: avg={run['avg_b']:.3f}B; {pieces}")
    for model in ([] if SIMPLE_STAGE2 else ["MobileNetV2", "AlexNet"]):
        model_runs = [run for run in scaling_runs if run["model"] == model]
        baseline = next(
            (run for run in model_runs if run["core"] == "4" and run["avg_b"] > 0),
            None,
        )
        if baseline is None:
            continue
        base = baseline["avg_b"]
        pieces = ", ".join(
            f"{run['core']}core={run['avg_b'] / base:.3f} ({base / run['avg_b']:.2f}x)"
            for run in model_runs
        )
        print(f"{model} Rocket scaling: {pieces}")


if __name__ == "__main__":
    main()

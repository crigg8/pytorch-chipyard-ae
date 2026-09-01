#!/usr/bin/env python3
"""Record and summarize the sampled-kernel RTL-turnaround experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RAW_FIELDS = [
    "run_id", "timestamp_utc", "trial", "kernel_id", "kernel_label",
    "source_model", "shape", "macs", "toolchain", "phase", "simulator",
    "total_wall_s", "status", "exit_code", "artifact_path", "log_path",
    "target_metadata", "notes",
]

SUMMARY_FIELDS = [
    "kernel_id", "kernel", "source_model", "shape", "macs",
    "pytorch_compile_s", "firesim_s", "tvm_compile_s", "verilator_s",
    "pytorch_compile_trials", "firesim_trials", "tvm_compile_trials",
    "verilator_trials", "status",
]

MEASUREMENT_FIELDS = [
    "pytorch_compile_s", "firesim_s", "tvm_compile_s", "verilator_s",
]

TRIAL_FIELDS = [
    "pytorch_compile_trials", "firesim_trials", "tvm_compile_trials",
    "verilator_trials",
]

TVM_COMPILE_SCOPE = (
    "gemmini.preprocess_pass through relay.build, MLF export, "
    "and microTVM project generation"
)
TVM_COMPILE_EXCLUDES = ["TFLite model preparation", "target C/ELF build"]


TABLE4_EXPERIMENT: dict[str, Any] = {
    "purpose": (
        "Bounded cycle-accurate turnaround experiment; not an "
        "inference-performance comparison."
    ),
    "selection_policy": (
        "One operation from each of three evaluated CNNs, restricted to "
        "1.28-3.10 million MACs so that the Verilator reference remains feasible."
    ),
    "kernels": [
        {
            "id": "squeezenet_fire2_squeeze",
            "label": "SqueezeNet fire2 squeeze",
            "source_model": "SqueezeNet 1.1",
            "source_operation": "fire2 squeeze 1x1 convolution lowered to GEMM",
            "m": 3025,
            "n": 16,
            "k": 64,
            "macs": 3097600,
        },
        {
            "id": "resnet50_classifier",
            "label": "ResNet-50 classifier",
            "source_model": "ResNet-50",
            "source_operation": "final fully connected layer",
            "m": 1,
            "n": 1000,
            "k": 2048,
            "macs": 2048000,
        },
        {
            "id": "mobilenetv2_classifier",
            "label": "MobileNetV2 classifier",
            "source_model": "MobileNetV2",
            "source_operation": "final fully connected layer",
            "m": 1,
            "n": 1000,
            "k": 1280,
            "macs": 1280000,
        },
    ],
}


def table4_kernels() -> list[dict[str, Any]]:
    kernels = TABLE4_EXPERIMENT["kernels"]
    seen: set[str] = set()
    for kernel in kernels:
        kernel_id = kernel.get("id")
        if not kernel_id or kernel_id in seen:
            raise SystemExit(f"missing or duplicate built-in kernel id: {kernel_id!r}")
        seen.add(kernel_id)
        computed = int(kernel["m"]) * int(kernel["n"]) * int(kernel["k"])
        if computed != int(kernel["macs"]):
            raise SystemExit(f"incorrect MAC count for {kernel_id}: {kernel['macs']} != {computed}")
    return kernels


def kernel_map() -> dict[str, dict[str, Any]]:
    return {kernel["id"]: kernel for kernel in table4_kernels()}


def get_kernel(kernel_id: str) -> dict[str, Any]:
    kernel = kernel_map().get(kernel_id)
    if kernel is None:
        choices = ", ".join(item["id"] for item in table4_kernels())
        raise SystemExit(f"unknown kernel {kernel_id!r}; expected one of: {choices}")
    return kernel


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def init_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as csv_file:
            header = next(csv.reader(csv_file), [])
        if header != RAW_FIELDS:
            raise SystemExit(f"unexpected CSV header in {path}: {header}")
        return
    with path.open("w", newline="") as csv_file:
        csv.DictWriter(csv_file, fieldnames=RAW_FIELDS).writeheader()


def append_row(args: argparse.Namespace) -> None:
    path = Path(args.csv)
    init_csv(path)
    row = {
        "run_id": args.run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "trial": args.trial,
        "kernel_id": args.kernel_id,
        "kernel_label": args.kernel_label,
        "source_model": args.source_model,
        "shape": args.shape,
        "macs": args.macs,
        "toolchain": args.toolchain,
        "phase": args.phase,
        "simulator": args.simulator,
        "total_wall_s": format_float(float(args.total_wall_s)) if args.total_wall_s else "",
        "status": args.status,
        "exit_code": args.exit_code,
        "artifact_path": args.artifact_path,
        "log_path": args.log_path,
        "target_metadata": args.target_metadata,
        "notes": args.notes,
    }
    with path.open("a", newline="") as csv_file:
        csv.DictWriter(csv_file, fieldnames=RAW_FIELDS).writerow(row)


def latest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row["kernel_id"], row["trial"], row["toolchain"],
            row["phase"], row["simulator"],
        )
        latest[key] = row
    return list(latest.values())


def median(rows: list[dict[str, str]]) -> float | None:
    values = [float(row["total_wall_s"]) for row in rows if row.get("total_wall_s")]
    return statistics.median(values) if values else None


def measurement_rows(
    rows: list[dict[str, str]], toolchain: str, phase: str, simulator: str
) -> list[dict[str, str]]:
    return [
        row for row in rows
        if row["status"] == "PASS" and row["toolchain"] == toolchain
        and row["phase"] == phase and row["simulator"] == simulator
    ]


def summarize(raw_path: Path, output_path: Path) -> None:
    with raw_path.open(newline="") as csv_file:
        active_rows = latest_rows(list(csv.DictReader(csv_file)))
    summaries: list[dict[str, str]] = []
    for kernel in table4_kernels():
        rows = [row for row in active_rows if row["kernel_id"] == kernel["id"]]
        pc_compile = measurement_rows(rows, "PyTorch-Chipyard", "compile", "")
        firesim = measurement_rows(rows, "PyTorch-Chipyard", "rtl", "firesim")
        tvm_compile = measurement_rows(rows, "TVM-Gemmini", "compile", "")
        verilator = measurement_rows(rows, "TVM-Gemmini", "rtl", "verilator")
        expected = [pc_compile, firesim, tvm_compile, verilator]
        if any(row["status"] == "FAIL" for row in rows):
            status = "FAIL"
        elif all(group for group in expected):
            status = "PASS"
        elif rows:
            status = "PARTIAL"
        else:
            status = "MISSING"
        summaries.append({
            "kernel_id": kernel["id"],
            "kernel": kernel["label"],
            "source_model": kernel["source_model"],
            "shape": f"{kernel['m']}x{kernel['n']}x{kernel['k']}",
            "macs": str(kernel["macs"]),
            "pytorch_compile_s": format_float(median(pc_compile)),
            "firesim_s": format_float(median(firesim)),
            "tvm_compile_s": format_float(median(tvm_compile)),
            "verilator_s": format_float(median(verilator)),
            "pytorch_compile_trials": str(len(pc_compile)),
            "firesim_trials": str(len(firesim)),
            "tvm_compile_trials": str(len(tvm_compile)),
            "verilator_trials": str(len(verilator)),
            "status": status,
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)


def write_latex(summary_path: Path, output_path: Path) -> None:
    with summary_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    def cell(value: str) -> str:
        return f"{float(value):.1f}\\,s" if value else "N/A"

    lines = ["% Generated by scripts/table4_results.py; do not edit by hand."]
    for row in rows:
        label = row["kernel"].replace("_", r"\_")
        shape = row["shape"].replace("x", r"\!\times\!")
        lines.append(
            rf"\makecell[c]{{{label}\\(${shape}$)}} & {cell(row['pytorch_compile_s'])} & {cell(row['firesim_s'])} & {cell(row['tvm_compile_s'])} & {cell(row['verilator_s'])} \\"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def validate_summary(summary_path: Path, minimum_trials: int = 1) -> None:
    if not summary_path.is_file() or summary_path.stat().st_size == 0:
        raise SystemExit(f"Table 4 summary is missing or empty: {summary_path}")
    with summary_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != SUMMARY_FIELDS:
            raise SystemExit(
                f"unexpected Table 4 CSV header in {summary_path}: {reader.fieldnames}"
            )
        rows = list(reader)

    expected_ids = [str(kernel["id"]) for kernel in table4_kernels()]
    actual_ids = [row["kernel_id"] for row in rows]
    if actual_ids != expected_ids:
        raise SystemExit(
            f"Table 4 kernel rows do not match the experiment: {actual_ids} != {expected_ids}"
        )

    for row in rows:
        kernel_id = row["kernel_id"]
        if row["status"] != "PASS":
            raise SystemExit(
                f"Table 4 row {kernel_id} is not complete: status={row['status']}"
            )
        for field in MEASUREMENT_FIELDS:
            try:
                value = float(row[field])
            except (TypeError, ValueError):
                raise SystemExit(
                    f"Table 4 row {kernel_id} has no numeric {field}: {row[field]!r}"
                ) from None
            if value <= 0:
                raise SystemExit(
                    f"Table 4 row {kernel_id} has non-positive {field}: {value}"
                )
        for field in TRIAL_FIELDS:
            try:
                value = int(row[field])
            except (TypeError, ValueError):
                raise SystemExit(
                    f"Table 4 row {kernel_id} has invalid {field}: {row[field]!r}"
                ) from None
            if value < minimum_trials:
                raise SystemExit(
                    f"Table 4 row {kernel_id} has {value} {field}; expected at least {minimum_trials}"
                )


def has_passed_measurement(
    path: Path, *, trial: str, kernel_id: str, toolchain: str,
    phase: str, simulator: str,
) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="") as csv_file:
        active = latest_rows(list(csv.DictReader(csv_file)))
    return any(
        row["trial"] == trial and row["kernel_id"] == kernel_id
        and row["toolchain"] == toolchain and row["phase"] == phase
        and row["simulator"] == simulator and row["status"] == "PASS"
        for row in active
    )


def has_current_tvm_compile_metadata(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        metadata = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("schema_version") == 2
        and metadata.get("compile_scope") == TVM_COMPILE_SCOPE
        and metadata.get("compile_excludes") == TVM_COMPILE_EXCLUDES
    )


def extract_last(path: Path, key: str) -> str:
    matches = re.findall(
        rf"(?:^|\[compile\]\s+){re.escape(key)}=([0-9]+(?:\.[0-9]+)?)",
        path.read_text(errors="replace"), re.MULTILINE,
    )
    if not matches:
        raise SystemExit(f"{key}=... not found in {path}")
    return matches[-1]


def extract_firesim_wall(path: Path) -> str:
    text = path.read_text(errors="replace")
    marker = re.findall(
        r"^TABLE4_FIRESIM_WALL_S=([0-9]+(?:\.[0-9]+)?)$", text, re.MULTILINE
    )
    if marker:
        return marker[-1]
    if "*** PASSED ***" not in text:
        raise SystemExit(f"FireSim PASS marker not found in {path}")
    summary = re.findall(
        r"^Wallclock Time Elapsed:\s*([0-9]+(?:\.[0-9]+)?)\s*s\s*$",
        text, re.MULTILINE,
    )
    if not summary:
        raise SystemExit(f"FireSim wallclock summary not found in {path}")
    return summary[-1]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--csv", required=True)
    append = commands.add_parser("append")
    append.add_argument("--csv", required=True)
    append.add_argument("--run-id", required=True)
    append.add_argument("--trial", required=True)
    append.add_argument("--kernel-id", required=True)
    append.add_argument("--kernel-label", required=True)
    append.add_argument("--source-model", required=True)
    append.add_argument("--shape", required=True)
    append.add_argument("--macs", required=True)
    append.add_argument("--toolchain", choices=["PyTorch-Chipyard", "TVM-Gemmini"], required=True)
    append.add_argument("--phase", choices=["compile", "rtl"], required=True)
    append.add_argument("--simulator", choices=["", "firesim", "verilator"], default="")
    append.add_argument("--total-wall-s", default="")
    append.add_argument("--status", choices=["PASS", "FAIL", "SKIP"], required=True)
    append.add_argument("--exit-code", default="")
    append.add_argument("--artifact-path", default="")
    append.add_argument("--log-path", default="")
    append.add_argument("--target-metadata", default="")
    append.add_argument("--notes", default="")
    summary = commands.add_parser("summarize")
    summary.add_argument("--csv", required=True)
    summary.add_argument("--output", required=True)
    latex = commands.add_parser("latex")
    latex.add_argument("--summary", required=True)
    latex.add_argument("--output", required=True)
    validate = commands.add_parser("validate-summary")
    validate.add_argument("--summary", required=True)
    validate.add_argument("--minimum-trials", type=int, default=1)
    commands.add_parser("list-kernels")
    field = commands.add_parser("kernel-field")
    field.add_argument("--kernel", required=True)
    field.add_argument("--field", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--log", required=True)
    extract.add_argument("--key", required=True)
    firesim = commands.add_parser("extract-firesim-wall")
    firesim.add_argument("--log", required=True)
    passed = commands.add_parser("has-pass")
    passed.add_argument("--csv", required=True)
    passed.add_argument("--trial", required=True)
    passed.add_argument("--kernel-id", required=True)
    passed.add_argument("--toolchain", required=True)
    passed.add_argument("--phase", choices=["compile", "rtl"], required=True)
    passed.add_argument("--simulator", choices=["", "firesim", "verilator"], default="")
    metadata = commands.add_parser("has-current-tvm-compile-metadata")
    metadata.add_argument("--metadata", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "init":
        init_csv(Path(args.csv))
    elif args.command == "append":
        append_row(args)
    elif args.command == "summarize":
        summarize(Path(args.csv), Path(args.output))
    elif args.command == "latex":
        write_latex(Path(args.summary), Path(args.output))
    elif args.command == "validate-summary":
        validate_summary(Path(args.summary), args.minimum_trials)
        print(f"TABLE4_CSV={Path(args.summary).resolve()}")
        print("TABLE4_STATUS=PASS")
    elif args.command == "list-kernels":
        for kernel in table4_kernels():
            print(kernel["id"])
    elif args.command == "kernel-field":
        kernel = get_kernel(args.kernel)
        if args.field not in kernel:
            raise SystemExit(f"unknown kernel/field: {args.kernel}/{args.field}")
        print(kernel[args.field])
    elif args.command == "extract":
        print(extract_last(Path(args.log), args.key))
    elif args.command == "extract-firesim-wall":
        print(extract_firesim_wall(Path(args.log)))
    elif args.command == "has-pass":
        raise SystemExit(0 if has_passed_measurement(
            Path(args.csv), trial=args.trial, kernel_id=args.kernel_id,
            toolchain=args.toolchain, phase=args.phase, simulator=args.simulator,
        ) else 1)
    elif args.command == "has-current-tvm-compile-metadata":
        raise SystemExit(
            0 if has_current_tvm_compile_metadata(Path(args.metadata)) else 1
        )


if __name__ == "__main__":
    main()

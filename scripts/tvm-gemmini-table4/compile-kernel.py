#!/usr/bin/env python3
"""Compile one model-derived dense operation with TVM-Gemmini."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import tarfile
import tempfile
import time
from typing import Any

import numpy as np
import tensorflow as tf
import tflite
import tvm
import tvm.contrib.gemmini as gemmini
from tvm import relay
from tvm.micro.testing.utils import create_header_file


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from table4_results import TVM_COMPILE_EXCLUDES, TVM_COMPILE_SCOPE, get_kernel
from smoke_test_spec import SMOKE_GEMM_KERNEL, smoke_gemm_kernel


SEED = 2027


def resolve_kernel(kernel_id: str) -> dict[str, Any]:
    if kernel_id == SMOKE_GEMM_KERNEL["id"]:
        return smoke_gemm_kernel()
    return get_kernel(kernel_id)


def load_gemmini_target() -> dict[str, Any]:
    include_dir = pathlib.Path(os.environ["TABLE4_TVM_GEMMINI_INCLUDE"]).resolve()
    params_path = include_dir / "gemmini_params.h"
    text = params_path.read_text()
    header_digest = hashlib.sha256()
    for header in sorted(include_dir.glob("*.h")):
        header_digest.update(header.name.encode())
        header_digest.update(b"\0")
        header_digest.update(header.read_bytes())

    def integer_define(name: str) -> int:
        match = re.search(rf"^#define {name} ([0-9]+)$", text, re.MULTILINE)
        if match is None:
            raise SystemExit(f"{name} was not found in {params_path}")
        return int(match.group(1))

    target = {
        "dtype": "int8",
        "accumulator_dtype": "int32",
        "gemmini_dim": integer_define("DIM"),
        "gemmini_bank_rows": integer_define("BANK_ROWS"),
        "gemmini_acc_rows": integer_define("ACC_ROWS"),
        "host": "rocket-singlecore",
        "simulator": "verilator",
        "gemmini_include": str(include_dir),
        "gemmini_params_sha256": hashlib.sha256(params_path.read_bytes()).hexdigest(),
        "gemmini_headers_sha256": header_digest.hexdigest(),
        "chipyard_commit": os.environ.get("TABLE4_TVM_CHIPYARD_COMMIT", ""),
    }
    if target["gemmini_dim"] != 16:
        raise SystemExit(f"expected DIM=16 in {params_path}")
    if not re.search(r"^typedef int8_t elem_t;$", text, re.MULTILINE):
        raise SystemExit(f"expected int8 elem_t in {params_path}")
    if not re.search(r"^typedef int32_t acc_t;$", text, re.MULTILINE):
        raise SystemExit(f"expected int32 acc_t in {params_path}")
    return target


def install_target_headers(project: pathlib.Path, include_dir: pathlib.Path) -> None:
    destination = project / "src" / "include"
    for source in sorted(include_dir.glob("*.h")):
        shutil.copy2(source, destination / source.name)


class DenseModel(tf.Module):
    def __init__(self, m: int, k: int, n: int, seed: int) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        self.weight = tf.Variable(
            rng.normal(0.0, 0.02, size=(k, n)).astype("float32"), name="weight"
        )
        # Keep an explicit, non-zero bias so TFLite retains the
        # qnn.dense -> nn.bias_add -> qnn.requantize pattern recognized by
        # this historical TVM-Gemmini integration.
        self.bias = tf.Variable(
            rng.normal(0.0, 0.01, size=(n,)).astype("float32"), name="bias"
        )
        self._m = m
        self._k = k

    @tf.function
    def matmul(self, x):
        return tf.linalg.matmul(x, self.weight) + self.bias

    def concrete_function(self):
        return self.matmul.get_concrete_function(
            tf.TensorSpec(shape=[self._m, self._k], dtype=tf.float32, name="input")
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help=(
            "stop after the timed Gemmini preprocess, Relay build, MLF export, "
            "and microTVM project-generation path"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_tflite(kernel: dict[str, Any]) -> bytes:
    kernel_seed = SEED + sum(ord(char) for char in str(kernel["id"]))
    model = DenseModel(
        int(kernel["m"]), int(kernel["k"]), int(kernel["n"]), kernel_seed
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [model.concrete_function()], model
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.int8
    converter._experimental_disable_per_channel = True

    def representative_data():
        rng = np.random.default_rng(kernel_seed + 1)
        for _ in range(4):
            yield [
                rng.normal(
                    0.0, 0.5, size=(int(kernel["m"]), int(kernel["k"]))
                ).astype("float32")
            ]

    converter.representative_dataset = representative_data
    return converter.convert()


def make_harness(project: pathlib.Path, kernel: dict[str, Any], output_len: int) -> None:
    generated_header = (project / "src" / "model" / "tvmgen_default.h").read_text()
    input_match = re.search(
        r"struct tvmgen_default_inputs \{\s*void\* ([A-Za-z0-9_]+);", generated_header
    )
    output_match = re.search(
        r"struct tvmgen_default_outputs \{\s*void\* ([A-Za-z0-9_]+);", generated_header
    )
    if input_match is None or output_match is None:
        raise RuntimeError("could not identify generated AOT input/output fields")
    input_field = input_match.group(1)
    output_field = output_match.group(1)
    harness = f'''#include <stdint.h>
#include <stdio.h>
#include "input.h"
#include "model/tvmgen_default.h"

static int8_t observed[{output_len}];

int main(void) {{
  printf("KERNEL_ID={kernel['id']}\\r\\n");
  printf("KERNEL_SHAPE={kernel['m']}x{kernel['n']}x{kernel['k']}\\r\\n");
  struct tvmgen_default_inputs inputs;
  struct tvmgen_default_outputs outputs;
  inputs.{input_field} = input;
  outputs.{output_field} = observed;
  int run_rc = tvmgen_default_run(&inputs, &outputs);
  printf("TVM_RUN_EXIT_CODE=%d\\r\\n", run_rc);
  printf("OUTPUT_COUNT={output_len}\\r\\n");
  printf("KERNEL_EXECUTION=%s\\r\\n", run_rc == 0 ? "PASS" : "FAIL");
  return run_rc;
}}
'''
    (project / "src" / "dense.c").write_text(harness)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    kernel = resolve_kernel(args.kernel)
    target_metadata = load_gemmini_target()
    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"refusing to overwrite {output_dir}; pass --force")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)
    tflite_buffer = build_tflite(kernel)
    interpreter = tf.lite.Interpreter(model_content=tflite_buffer)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    kernel_seed = SEED + sum(ord(char) for char in str(kernel["id"]))
    rng = np.random.default_rng(kernel_seed + 2)
    input_data = rng.integers(
        0, 256, size=tuple(int(value) for value in input_detail["shape"]), dtype=np.uint8
    )
    output_len = int(np.prod(output_detail["shape"]))

    flatbuffer = tflite.Model.GetRootAsModel(tflite_buffer, 0)
    shape_dict = {input_detail["name"]: tuple(int(value) for value in input_detail["shape"])}
    dtype_dict = {input_detail["name"]: "uint8"}
    gemmini.Environment.init_overwrite(
        dim=int(target_metadata["gemmini_dim"]),
        acc_rows=int(target_metadata["gemmini_acc_rows"]),
        bank_rows=int(target_metadata["gemmini_bank_rows"]),
    )
    mod, params = relay.frontend.from_tflite(
        flatbuffer, shape_dict=shape_dict, dtype_dict=dtype_dict
    )
    mod = relay.transform.InferType()(mod)
    if os.environ.get("TABLE4_DUMP_RELAY") == "1":
        print("RELAY_BEFORE_GEMMINI_BEGIN")
        print(mod)
        print("RELAY_BEFORE_GEMMINI_END")

    # Table 4 defines TVM compilation as the complete Gemmini-specific
    # model-to-project path. Model preparation and target C/ELF construction
    # remain outside this interval.
    compile_started = time.perf_counter()
    mod = gemmini.preprocess_pass(mod)
    runtime = relay.backend.Runtime("crt", {"system-lib": False})
    target = tvm.target.Target({"kind": "c", "device": "gemmini"})
    executor = relay.backend.Executor(
        "aot", options={"interface-api": "c", "unpacked-api": 1}
    )
    with gemmini.build_config(
        usmp_alg="hill_climb", opt_level=3, disabled_pass=["AlterOpLayout"]
    ):
        module = relay.build(
            mod, executor=executor, runtime=runtime, target=target, params=params
        )

    temp_dir = tvm.contrib.utils.tempdir()
    tvm.micro.export_model_library_format(module, temp_dir / "model.tar")
    with tempfile.NamedTemporaryFile() as extra_file:
        with tarfile.open(extra_file.name, "w:gz") as tar_file:
            create_header_file("input", input_data, "include/tvm", tar_file)
        template = pathlib.Path(tvm.micro.get_microtvm_template_projects("gemmini"))
        generated = tvm.micro.generate_project(
            template,
            module,
            output_dir,
            {"project_type": "dense_example", "extra_files_tar": extra_file.name},
        )
    compile_wall_s = time.perf_counter() - compile_started

    if os.environ.get("TABLE4_DUMP_RELAY") == "1":
        print("RELAY_AFTER_GEMMINI_BEGIN")
        print(mod)
        print("RELAY_AFTER_GEMMINI_END")

    generated_sources = list((output_dir / "src" / "model").glob("default_lib*.c"))
    if not any(
        "tiled_matmul_auto" in path.read_text(errors="replace")
        for path in generated_sources
    ):
        raise SystemExit("TVM output does not contain a Gemmini tiled_matmul_auto call")

    project_build_wall_s = None
    elf = None
    if not args.compile_only:
        # TVM-Gemmini vendors an older software snapshot. Compile the generated
        # program against the headers belonging to the Verilator RTL target so
        # the software-visible Gemmini parameters and command API cannot drift.
        install_target_headers(
            output_dir, pathlib.Path(str(target_metadata["gemmini_include"]))
        )
        make_harness(output_dir, kernel, output_len)
        build_started = time.perf_counter()
        generated.build()
        project_build_wall_s = time.perf_counter() - build_started
        elf = output_dir / "src" / "build" / "dense-baremetal"
        if not elf.is_file():
            raise SystemExit(f"generated project build did not produce {elf}")

    metadata = {
        "schema_version": 2,
        "kernel": kernel,
        "target": target_metadata,
        "compile_scope": TVM_COMPILE_SCOPE,
        "compile_excludes": TVM_COMPILE_EXCLUDES,
        "compile_only": args.compile_only,
        "compile_wall_s": compile_wall_s,
        "project_build_wall_s": project_build_wall_s,
        "elf": str(elf) if elf is not None else None,
    }
    (output_dir / "table4-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"KERNEL_ID={kernel['id']}")
    print(f"KERNEL_LABEL={kernel['label']}")
    print(f"KERNEL_SHAPE={kernel['m']}x{kernel['n']}x{kernel['k']}")
    print(f"KERNEL_MACS={kernel['macs']}")
    print(
        "TARGET_METADATA="
        f"int8,gemmini-dim{target_metadata['gemmini_dim']},"
        f"acc-rows{target_metadata['gemmini_acc_rows']},"
        "rocket-singlecore,verilator"
    )
    print(f"COMPILE_WALL_S={compile_wall_s:.6f}")
    print(f"COMPILE_ONLY={int(args.compile_only)}")
    if project_build_wall_s is not None:
        print(f"PROJECT_BUILD_WALL_S={project_build_wall_s:.6f}")
    if elf is not None:
        print(f"ELF={elf}")


if __name__ == "__main__":
    main()

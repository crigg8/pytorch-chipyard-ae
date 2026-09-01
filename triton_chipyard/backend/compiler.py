from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import re
import shutil
import subprocess
import functools
import sys
import triton
from pathlib import Path

from .custom_linalg import load_custom_linalg_config

def _parse_int(name: str, value: Any) -> int:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if raw == "":
        return default
    return _parse_int(name, raw)


def _get_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean-like value, got {raw!r}")


def _use_gemmini() -> bool:
    return _get_bool_env("TRITON_CHIPYARD_USE_GEMMINI", False)


def _use_rvv() -> bool:
    return _get_bool_env("TRITON_CHIPYARD_USE_RVV", True)


def _enable_alias_first() -> bool:
    return _get_bool_env("TRITON_CHIPYARD_ENABLE_ALIAS_FIRST", True)


def _get_riscv_varch() -> str:
    return os.getenv("TRITON_CHIPYARD_RISCV_VARCH", "").strip()


def _get_riscv_float_abi(mabi: str) -> str:
    return "hard" if mabi.endswith(("f", "d")) else "soft"


def _parse_riscv_varch(varch: str) -> Tuple[int, int]:
    match = re.fullmatch(r"vlen:(\d+),elen:(\d+)", varch.replace(" ", ""))
    if match is None:
        raise ValueError(
            "RISC-V VARCH must look like 'vlen:64,elen:32', "
            f"got {varch!r}."
        )

    vlen = int(match.group(1), 10)
    elen = int(match.group(2), 10)

    if vlen < 64 or vlen > 65536 or (vlen & (vlen - 1)) != 0:
        raise ValueError(
            f"RVV VLEN must be a power of two in [64, 65536] for the current buddy-llc, got {vlen}."
        )
    if elen not in (32, 64):
        raise ValueError(f"RVV ELEN must be 32 or 64 for FP lowering, got {elen}.")
    if elen > vlen:
        raise ValueError(f"RVV ELEN {elen} cannot exceed VLEN {vlen}.")

    return vlen, elen


def _get_riscv_vlen(varch: str = "") -> Optional[int]:
    if not varch:
        varch = _get_riscv_varch()
    if not varch:
        return None

    vlen, _ = _parse_riscv_varch(varch)
    return vlen


def _get_linalg_matmul_elem_bits(ttsharedir: str) -> Optional[int]:
    match = re.search(r"linalg\.matmul.*\b(f32|f64)\b", ttsharedir)
    if match is None:
        return None
    return int(match.group(1)[1:], 10)


def _get_rvv_vector_size(ttsharedir: str = "") -> int:
    raw = os.getenv("TRITON_CHIPYARD_RVV_VECTOR_SIZE", "")
    if raw != "":
        return _parse_int("TRITON_CHIPYARD_RVV_VECTOR_SIZE", raw)

    varch = _get_riscv_varch()
    if not varch:
        return 4

    vlen, elen = _parse_riscv_varch(varch)
    elem_bits = _get_linalg_matmul_elem_bits(ttsharedir)
    if elem_bits is None:
        elem_bits = elen
    elif elem_bits > elen:
        raise ValueError(
            f"RVV ELEN {elen} from {varch!r} cannot lower {elem_bits}-bit "
            "linalg.matmul operands."
        )

    if vlen % elem_bits != 0:
        raise ValueError(
            f"RVV VLEN {vlen} from {varch!r} must be divisible by element "
            f"width {elem_bits}."
        )

    return max(1, vlen // elem_bits)


def _get_riscv_target_features(march: str, varch: str = "") -> str:
    march = march.strip().lower()
    if not march.startswith(("rv32", "rv64")):
        raise ValueError(
            "TRITON_CHIPYARD_RISCV_MARCH must start with rv32 or rv64, "
            f"got {march!r}."
        )

    tail = march[4:]
    parts = [part for part in tail.split("_") if part]
    if not parts:
        raise ValueError(
            "TRITON_CHIPYARD_RISCV_MARCH must include at least one extension, "
            f"got {march!r}."
        )

    features = ["+buddyext"]
    base_ext = parts[0]
    base_feature_map = {
        "m": "+m",
        "a": "+a",
        "f": "+f",
        "d": "+d",
        "c": "+c",
        "v": "+v",
        "h": "+h",
    }
    for ext in base_ext:
        if ext == "i":
            continue
        if ext == "g":
            features.extend(["+m", "+a", "+f", "+d", "+zicsr", "+zifencei"])
            continue
        feature = base_feature_map.get(ext)
        if feature is None:
            raise ValueError(
                "Unsupported packed RISC-V extension "
                f"{ext!r} in TRITON_CHIPYARD_RISCV_MARCH={march!r}."
            )
        features.append(feature)

    for ext in parts[1:]:
        features.append(f"+{ext}")

    if varch:
        vlen, _ = _parse_riscv_varch(varch)
        features.append(f"+zvl{vlen}b")

    return ",".join(dict.fromkeys(features))


def _append_target_features(*feature_sets: str) -> str:
    features = []
    for feature_set in feature_sets:
        if not feature_set:
            continue
        for feature in feature_set.split(","):
            feature = feature.strip()
            if feature and feature not in features:
                features.append(feature)
    return ",".join(features)


def _get_gemmini_elem_acc_types(_ttsharedir: str) -> Tuple[str, str]:
    elem_t = os.getenv("TRITON_CHIPYARD_GEMMINI_ELEM_T", "i8")
    acc_t = os.getenv("TRITON_CHIPYARD_GEMMINI_ACC_T", "i32")

    return elem_t, acc_t


def _get_triton_chipyard_opt_path() -> str:
    path = os.getenv("TRITON_CHIPYARD_OPT_PATH", "")
    if path:
        return path

    raise Exception(
        "TRITON_CHIPYARD_OPT_PATH is not set. "
    )

def _get_buddy_mlir_path() -> str:
    path = os.getenv("BUDDY_BINARY_DIR", "")
    if path == "":
        raise Exception("BUDDY_BINARY_DIR is not set.")
    return path

def _is_perf_matmul_enabled() -> bool:
    raw = os.getenv("TRITON_CHIPYARD_PERF_OPS", "")
    if not raw:
        return False
    ops = {op.strip().lower() for op in re.split(r"[;,\s]+", raw) if op.strip()}
    return "matmul" in ops


def _has_fp_linalg_matmul(ttsharedir: str) -> bool:
    return bool(re.search(r"linalg\.matmul.*(f32|f64)", ttsharedir))

def _dump_ir_if_needed(files):
    global dump_cnt
    path = os.getenv("TRITON_CHIPYARD_DUMP_PATH", "")
    if not path:
        return
    os.makedirs(path, exist_ok=True)

    for f in files:
        base = os.path.basename(f)
        shutil.copy(f, os.path.join(path, base))

def _run_triton_chipyard_opt(args, *, pattern_context=""):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.stdout:
        sys.stdout.write(result.stdout)
        sys.stdout.flush()
    if result.stderr:
        sys.stderr.write(result.stderr)
        sys.stderr.flush()
    if result.returncode != 0 and pattern_context:
        raise subprocess.CalledProcessError(
            result.returncode,
            args,
            output=result.stdout,
            stderr=f"{pattern_context}\n{result.stderr}",
        )
    result.check_returncode()


def _ttir_to_ttsharedir(mod, metadata, options):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        use_custom_linalg = bool(options.custom_linalg_patterns)
        bufferized_path = os.path.join(tmpdir, "ttshared-bufferized.mlir")
        final_path = os.path.join(tmpdir, "ttshared.mlir")
        Path(src_path).write_text(ttir_code)
        _dump_ir_if_needed([src_path])
        triton_chipyard_opt_path = _get_triton_chipyard_opt_path()

        subprocess_args = [
            triton_chipyard_opt_path,
            src_path,
            (
                "--triton-to-linalg-experimental=enable-alias-first="
                + ("true" if _enable_alias_first() else "false")
            ),
            "--expand-rsqrt-for-chipyard",
            "--convert-math-to-libm",
        ]
        if _is_perf_matmul_enabled():
            subprocess_args.append("--perf-matmul")
        if use_custom_linalg:
            subprocess_args.extend(
                [
                    "--one-shot-bufferize=allow-return-allocs-from-loops",
                    "--buffer-deallocation-pipeline",
                ]
            )
        subprocess_args.extend(
            [
                "--mlir-print-debuginfo",
                "-o",
                bufferized_path if use_custom_linalg else final_path,
            ]
        )

        _run_triton_chipyard_opt(subprocess_args)
        if use_custom_linalg:
            _dump_ir_if_needed([bufferized_path])

        current_path = bufferized_path if use_custom_linalg else final_path
        for pattern_index, (target_op, function, operand_types) in enumerate(
            options.custom_linalg_patterns
        ):
            next_path = os.path.join(
                tmpdir, f"ttshared-custom-{pattern_index:03d}.mlir"
            )
            pass_options = f"target-op={target_op} function={function}"
            if operand_types:
                pass_options += f" operand-types={operand_types}"
            pattern_args = [
                triton_chipyard_opt_path,
                current_path,
                "--linalg-to-function-call=" + pass_options,
                "--mlir-print-debuginfo",
                "-o",
                next_path,
            ]
            context = (
                f"custom Linalg pattern {pattern_index}: "
                f"target_op={target_op!r}, function={function!r}, "
                f"operand_types={operand_types!r}"
            )
            _run_triton_chipyard_opt(pattern_args, pattern_context=context)
            _dump_ir_if_needed([next_path])
            current_path = next_path

        if current_path != final_path:
            shutil.copy2(current_path, final_path)
        _dump_ir_if_needed([final_path])

        if options.custom_linalg_library_path:
            metadata["triton_chipyard_custom_library_path"] = (
                options.custom_linalg_library_path
            )
            metadata["triton_chipyard_custom_library_sha256"] = (
                options.custom_linalg_library_sha256
            )
            metadata["triton_chipyard_custom_config_path"] = (
                options.custom_linalg_config_path
            )
            metadata["triton_chipyard_custom_config_sha256"] = (
                options.custom_linalg_config_sha256
            )
        return Path(final_path).read_text()


def _optimize_ttsharedir(ttsharedir: str):
    return ttsharedir

def _ttsharedir_to_buddymlir(ttsharedir: str, metadata, options):
    pattern = r'func\.func\s+@([^\s(]+)' 
    matches = re.findall(pattern, ttsharedir)
    assert len(matches) == 1, matches
    metadata["name"] = matches[0]

    metadata_str = (
        f'"ttc.gemmini_tile_i" = {metadata["gemmini_tile_i"]}, '
        f'"ttc.gemmini_tile_j" = {metadata["gemmini_tile_j"]}, '
        f'"ttc.gemmini_tile_k" = {metadata["gemmini_tile_k"]} '
    )
    ttsharedir = ttsharedir.replace("module {", f'module attributes {{{metadata_str}}}\n {{')

    BUDDY_BINARY_DIR = _get_buddy_mlir_path()
    BUDDY_OPT_PATH = os.path.join(BUDDY_BINARY_DIR, "buddy-opt")
    BUDDY_TRANSLATE_PATH = os.path.join(BUDDY_BINARY_DIR, "buddy-translate")
    BUDDY_LLC_PATH = os.path.join(BUDDY_BINARY_DIR, "buddy-llc")
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "ttshared.mlir")
        temp_path = os.path.join(tmpdir, "temp.mlir")
        dst_path = os.path.join(tmpdir, "log-ttshared.o")
        Path(src_path).write_text(ttsharedir)
        addr_len = _parse_int("metadata.addr_len", metadata.get("addr_len", 32))
        dim = _parse_int("metadata.dim", metadata.get("dim", 16))
        bank_rows = _parse_int("metadata.bank_rows", metadata.get("bank_rows", 4096))
        acc_rows = _parse_int("metadata.acc_rows", metadata.get("acc_rows", 1024))
        use_gemmini = _use_gemmini()
        use_rvv = _use_rvv()
        rvv_vector_size = _get_rvv_vector_size(ttsharedir) if use_rvv and not use_gemmini else 0
        march = os.getenv("TRITON_CHIPYARD_RISCV_MARCH", "").strip()
        mabi = os.getenv("TRITON_CHIPYARD_RISCV_MABI", "").strip()
        varch = _get_riscv_varch() if use_rvv and not use_gemmini else ""
        rvv_vlen = _get_riscv_vlen(varch) if varch else None
        float_abi = _get_riscv_float_abi(mabi) if mabi else ""
        target_features = _get_riscv_target_features(march, varch) if march else ""
        if use_rvv and not use_gemmini:
            target_features = _append_target_features(
                target_features, "+dlen-factor-2"
            )

        elem_t, acc_t = _get_gemmini_elem_acc_types(ttsharedir)
        lower_gemmini_args = (
            f"addr_len={addr_len} "
            f"dim={dim} "
            f"bank_rows={bank_rows} "
            f"acc_rows={acc_rows} "
            f"elem_t={elem_t} "
            f"acc_t={acc_t}"
        )
        convert_linalg_to_gemmini_args = (
            f"elem_t={elem_t} "
            f"acc_t={acc_t}"
        )

        elem_t, acc_t = _get_gemmini_elem_acc_types(ttsharedir)
        lower_gemmini_args = (
            f"addr_len={addr_len} "
            f"dim={dim} "
            f"bank_rows={bank_rows} "
            f"acc_rows={acc_rows} "
            f"elem_t={elem_t} "
            f"acc_t={acc_t}"
        )
        convert_linalg_to_gemmini_args = (
            f"elem_t={elem_t} "
            f"acc_t={acc_t}"
        )

        buddy_opt_args = [
            BUDDY_OPT_PATH,
            src_path,
        ]
        if not options.custom_linalg_patterns:
            buddy_opt_args.extend(
                [
                    '--one-shot-bufferize=allow-return-allocs-from-loops',
                    '--buffer-deallocation-pipeline',
                ]
            )
        if use_gemmini:
            buddy_opt_args.extend([
                f'--convert-linalg-to-gemmini={convert_linalg_to_gemmini_args}',
                '--expand-strided-metadata',
                '--convert-linalg-to-loops',
                f'--lower-gemmini={lower_gemmini_args}',
            ])
        elif use_rvv:
            buddy_opt_args.extend([
                f'--matmul-vectorization=vector-type=scalable vector-size={rvv_vector_size}',
                '--expand-strided-metadata',
                '--convert-linalg-to-loops',
                '--convert-scf-to-cf',
                '--convert-vector-to-llvm',
                f'--lower-gemmini={lower_gemmini_args}',
                '--reconcile-unrealized-casts',
            ])
        else:
            buddy_opt_args.extend([
                '--expand-strided-metadata',
                '--convert-linalg-to-loops',
                '--convert-scf-to-cf',
                '--convert-vector-to-llvm',
                f'--lower-gemmini={lower_gemmini_args}',
                '--reconcile-unrealized-casts',
            ])
        buddy_opt_args.extend(['-o', temp_path])
        buddy_opt_result = subprocess.run(
            buddy_opt_args,
            capture_output=True,
            text=True,
        )
        if buddy_opt_result.stdout:
            sys.stdout.write(buddy_opt_result.stdout)
            sys.stdout.flush()
        if buddy_opt_result.stderr:
            sys.stderr.write(buddy_opt_result.stderr)
            sys.stderr.flush()
        buddy_opt_result.check_returncode()
        target_abi_flag = f"-target-abi {mabi}" if mabi else ""
        target_features_flag = f"-mattr={target_features}" if target_features else ""
        rvv_vector_bits_flags = (
            f"-riscv-v-vector-bits-min={rvv_vlen} "
            f"-riscv-v-vector-bits-max={rvv_vlen}"
            if rvv_vlen is not None
            else ""
        )
        float_abi_flag = f"-float-abi={float_abi}" if float_abi else ""
        cmd = f"""
"{BUDDY_TRANSLATE_PATH}" --buddy-to-llvmir "{temp_path}" | \
"{BUDDY_LLC_PATH}" -filetype=obj -mtriple=riscv64 \
{target_abi_flag} {target_features_flag} {rvv_vector_bits_flags} \
{float_abi_flag} --opt-disable=riscv-vector-peephole -o "{dst_path}"
"""
        subprocess.check_call(["bash", "-lc", cmd])
        _dump_ir_if_needed([temp_path, dst_path])

        return Path(dst_path).read_bytes()

def _optimize_buddy_mlir(buddy_mlir: str):
    return buddy_mlir


@dataclass(frozen=True)
class ChipyardOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    supported_fp8_dtypes: Tuple[str] = ()
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    sanitize_overflow: bool = True
    addr_len: int = 32
    dim: int = 16
    bank_rows: int = 4096
    acc_rows: int = 1024
    gemmini_tile_i: int = 0
    gemmini_tile_j: int = 0
    gemmini_tile_k: int = 0
    custom_linalg_config_path: str = ""
    custom_linalg_config_sha256: str = ""
    custom_linalg_library_path: str = ""
    custom_linalg_library_sha256: str = ""
    custom_linalg_patterns: Tuple[Tuple[str, str, str], ...] = ()

    def __post_init__(self):
        pass

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name,
                       val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class ChipyardBackend(BaseBackend):
    binary_ext = 'obj'

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'chipyard'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        custom_linalg = load_custom_linalg_config()
        args = {
            "arch": self.target.arch,
            "addr_len": _get_int_env("TRITON_CHIPYARD_GEMMINI_ADDR_LEN", ChipyardOptions.addr_len),
            "dim": _get_int_env("TRITON_CHIPYARD_GEMMINI_DIM", ChipyardOptions.dim),
            "bank_rows": _get_int_env("TRITON_CHIPYARD_GEMMINI_BANK_ROWS", ChipyardOptions.bank_rows),
            "acc_rows": _get_int_env("TRITON_CHIPYARD_GEMMINI_ACC_ROWS", ChipyardOptions.acc_rows),
            "gemmini_tile_i": ChipyardOptions.gemmini_tile_i,
            "gemmini_tile_j": ChipyardOptions.gemmini_tile_j,
            "gemmini_tile_k": ChipyardOptions.gemmini_tile_k,
        }
        args.update(
            {k: opts[k] for k in ChipyardOptions.__dataclass_fields__.keys() if k in opts and opts[k] is not None}
        )
        args.update(
            {
                "custom_linalg_config_path": (
                    custom_linalg.config_path if custom_linalg is not None else ""
                ),
                "custom_linalg_config_sha256": (
                    custom_linalg.config_sha256 if custom_linalg is not None else ""
                ),
                "custom_linalg_library_path": (
                    custom_linalg.library_path if custom_linalg is not None else ""
                ),
                "custom_linalg_library_sha256": (
                    custom_linalg.library_sha256 if custom_linalg is not None else ""
                ),
                "custom_linalg_patterns": (
                    tuple(
                        (
                            pattern.target_op,
                            pattern.function,
                            pattern.cli_operand_types(),
                        )
                        for pattern in custom_linalg.patterns
                    )
                    if custom_linalg is not None
                    else ()
                ),
            }
        )
        for int_key in (
            "addr_len",
            "dim",
            "bank_rows",
            "acc_rows",
            "gemmini_tile_i",
            "gemmini_tile_j",
            "gemmini_tile_k",
        ):
            args[int_key] = _parse_int(int_key, args[int_key])
        return ChipyardOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name,
            metadata.gemmini_tile_i,
            metadata.gemmini_tile_j,
            metadata.gemmini_tile_k,
            metadata.custom_linalg_library_path,
            metadata.custom_linalg_library_sha256,
        )

    def load_dialects(self, ctx):
        return

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        passes.common.add_cse(pm)
        pm.run(mod)
        return mod

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: self.make_ttir(
            src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: _optimize_ttsharedir(
            _ttir_to_ttsharedir(src, metadata, options))
        stages["obj"] = lambda src, metadata: _optimize_buddy_mlir(
            _ttsharedir_to_buddymlir(src, metadata, options))

    @functools.lru_cache()
    def hash(self):
        return self.target

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

import hashlib
import sysconfig

import os, struct, subprocess, tempfile, platform, re
import importlib.util
import sys
import math

from pathlib import Path

from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

dirname = os.path.dirname(os.path.realpath(__file__))
_LAST_KERNEL_CYCLES = None
_LAST_MATMUL_CYCLES = None
PACKET_MAX_DIMS = 5


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if raw == "":
        return default
    return int(raw, 0)


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


def _get_chipyard_env_path() -> str:
    path = os.getenv("CHIPYARD_ENV_PATH", "")
    if path == "":
        raise Exception("CHIPYARD_ENV_PATH is not set.")
    return path
  
def _get_llvm_project_path() -> str:
    path = os.getenv("LLVM_PROJECT_PATH", "")
    if path == "":
        raise Exception("LLVM_PROJECT_PATH is not set")
    return path

def _is_perf_matmul_enabled() -> bool:
    raw = os.getenv("TRITON_CHIPYARD_PERF_OPS", "")
    if not raw:
        return False
    ops = {op.strip().lower() for op in re.split(r"[;,\s]+", raw) if op.strip()}
    return "matmul" in ops

def _get_packet_tmpdir() -> str:
    # Prefer tmpfs for packet I/O if available. This reduces host-side file I/O latency.
    preferred = os.getenv("TRITON_CHIPYARD_PACKET_TMPDIR", "/dev/shm")
    if os.path.isdir(preferred) and os.access(preferred, os.W_OK):
        return preferred
    return tempfile.gettempdir()

def _read_packet_perf_counters(packet_path):
    # Perf counters are written back by guest runtime:
    # - kernel cycles: entries[0].reserved1
    # - matmul cycles: entries[1].reserved1 (when perf matmul is enabled)
    MAX_DIMS = PACKET_MAX_DIMS
    ENTRY_STRUCT = struct.Struct(
        "<" +
        "6I" +
        f"{MAX_DIMS}Q" +
        f"{MAX_DIMS}q" +
        "3Q"
    )

    with open(packet_path, "rb") as f:
        header = f.read(32)
        if len(header) != 32:
            return None, None
        _magic, _version, n_args, entry_size, _reserved, _payload_offset = struct.unpack("<8sIIIIQ", header)
        if n_args == 0 or entry_size != ENTRY_STRUCT.size:
            return None, None
        eb = f.read(entry_size)
        if len(eb) != entry_size:
            return None, None
        first = ENTRY_STRUCT.unpack(eb)
        kernel_cycles = int(first[-1])
        matmul_cycles = None

        if n_args > 1:
            eb = f.read(entry_size)
            if len(eb) == entry_size:
                second = ENTRY_STRUCT.unpack(eb)
                matmul_cycles = int(second[-1])

        return kernel_cycles, matmul_cycles

def _read_packet_cycles(packet_path):
    kernel_cycles, _ = _read_packet_perf_counters(packet_path)
    return kernel_cycles

# -------------------- Launcher ----------------------------
def _ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    if ty == "constexpr":
        return "pyobject*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        # Use MLIR runtime half type for memref descriptors.
        "fp16": "f16",
        "bf16": "bf16",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def _tensor_elem_ty_to_cpp(ty):
    if ty in ("i1", "u1", "bool"):
        return "uint8_t"
    return _ty_to_cpp(ty)


def _scalar_packet_get_expr(ty: str, entry_expr: str, payload_expr: str) -> str:
    # Kernel signatures use aliases like fp32/fp64 while runtime helpers are
    # packet_get_f32/f64, so normalize here.
    if ty in ("fp32", "f32"):
        getter = "packet_get_f32"
    elif ty == "bf16":
        getter = "packet_get_f32"
    elif ty in ("fp64", "f64"):
        getter = "packet_get_f64"
    elif ty in ("i64", "u64"):
        getter = "packet_get_i64"
    elif ty in ("i32", "u32", "i16", "u16", "i8", "u8", "i1", "u1"):
        getter = "packet_get_i32"
    elif ty == "bool":
        getter = "packet_get_bool"
    else:
        raise KeyError(f"Unsupported scalar signature type for packet getter: {ty}")

    c_ty = _ty_to_cpp(ty)
    return f"static_cast<{c_ty}>({getter}({entry_expr}, {payload_expr}))"

def _write_packet_bin(args, packet_path):
    import ctypes
    import torch
    import numpy as np

    MAGIC = b"TTCIPKT1"
    VERSION = 1
    MAX_DIMS = PACKET_MAX_DIMS

    # torch dtype -> enum (너 마음대로 확장 가능)
    TORCH_DTYPE_ENUM = {
        torch.float32: 1,
        torch.float16: 2,
        torch.bfloat16: 3,
        torch.float64: 4,
        torch.int32: 5,
        torch.int64: 6,
        torch.int16: 7,
        torch.int8: 8,
        torch.uint8: 9,
        torch.bool: 10,
    }

    KIND_SCALAR = 0
    KIND_TENSOR = 1

    SCALAR_INT64 = 1
    SCALAR_FLOAT64 = 2
    SCALAR_BOOL = 3
    SCALAR_BYTES = 4

    # Arg entry layout (little-endian)
    # u32 kind, u32 scalar_type, u32 tensor_dtype, u32 ndim, u32 flags, u32 reserved0
    # u64 sizes[MAX_DIMS]
    # i64 strides[MAX_DIMS]
    # u64 nbytes, u64 data_offset, u64 reserved1
    # For tensor entries, storage_offset (in elements) is split across:
    #   flags     = low 32 bits
    #   reserved0 = high 32 bits
    # reserved1 remains available for runtime writeback metadata.
    ENTRY_STRUCT = struct.Struct(
        "<" +
        "6I" +              # 6 * u32
        f"{MAX_DIMS}Q" +    # sizes
        f"{MAX_DIMS}q" +    # strides
        "3Q"                # nbytes, data_offset, reserved1
    )
    entry_size = ENTRY_STRUCT.size

    entries = []
    payload_chunks = []

    def _as_tensor_payload(t):
        t2 = t.detach()
        if t2.device.type != "cpu":
            t2 = t2.cpu()
        storage = t2.untyped_storage()
        nbytes = int(storage.nbytes())
        if nbytes:
            b = ctypes.string_at(storage.data_ptr(), nbytes)
        else:
            b = b""
        return t2, b, int(t2.storage_offset())

    # First pass: build entry metadata and payload chunks (offsets filled later)
    for a in args:
        import torch
        if isinstance(a, torch.Tensor):
            t2, b, storage_offset = _as_tensor_payload(a)
            dtype_enum = TORCH_DTYPE_ENUM.get(t2.dtype, 0)
            if dtype_enum == 0:
                raise TypeError(f"Unsupported torch dtype for packet: {t2.dtype}")

            shape = list(t2.shape)
            stride = list(t2.stride())

            ndim = len(shape)
            if ndim > MAX_DIMS:
                raise ValueError(f"Tensor ndim {ndim} > {MAX_DIMS} not supported in packet v1")

            sizes = shape + [0] * (MAX_DIMS - ndim)
            strides = stride + [0] * (MAX_DIMS - ndim)

            payload_chunks.append(b)
            storage_offset_lo = storage_offset & 0xFFFFFFFF
            storage_offset_hi = (storage_offset >> 32) & 0xFFFFFFFF
            entries.append({
                "kind": KIND_TENSOR,
                "scalar_type": 0,
                "tensor_dtype": dtype_enum,
                "ndim": ndim,
                "flags": storage_offset_lo,
                "sizes": sizes,
                "strides": strides,
                "nbytes": len(b),
                "data_offset": 0,  # fill later
                "reserved0": storage_offset_hi,
                "reserved1": 0,
            })

        elif isinstance(a, bool):
            # scalar bool -> 8 bytes payload (u64)
            b = struct.pack("<Q", 1 if a else 0)
            payload_chunks.append(b)
            entries.append({
                "kind": KIND_SCALAR,
                "scalar_type": SCALAR_BOOL,
                "tensor_dtype": 0,
                "ndim": 0,
                "flags": 0,
                "reserved0": 0,
                "sizes": [0] * MAX_DIMS,
                "strides": [0] * MAX_DIMS,
                "nbytes": 8,
                "data_offset": 0,
                "reserved1": 0,
            })

        elif isinstance(a, int):
            b = struct.pack("<q", a)
            payload_chunks.append(b)
            entries.append({
                "kind": KIND_SCALAR,
                "scalar_type": SCALAR_INT64,
                "tensor_dtype": 0,
                "ndim": 0,
                "flags": 0,
                "reserved0": 0,
                "sizes": [0] * MAX_DIMS,
                "strides": [0] * MAX_DIMS,
                "nbytes": 8,
                "data_offset": 0,
                "reserved1": 0,
            })

        elif isinstance(a, float):
            b = struct.pack("<d", a)
            payload_chunks.append(b)
            entries.append({
                "kind": KIND_SCALAR,
                "scalar_type": SCALAR_FLOAT64,
                "tensor_dtype": 0,
                "ndim": 0,
                "flags": 0,
                "reserved0": 0,
                "sizes": [0] * MAX_DIMS,
                "strides": [0] * MAX_DIMS,
                "nbytes": 8,
                "data_offset": 0,
                "reserved1": 0,
            })
        
        elif isinstance(a, str):
            b = a.encode("utf-8")
            payload_chunks.append(b)
            entries.append({
                "kind": KIND_SCALAR,
                "scalar_type": SCALAR_BYTES,
                "tensor_dtype": 0,
                "ndim": 0,
                "flags": 0,
                "reserved0": 0,
                "sizes": [0] * MAX_DIMS,
                "strides": [0] * MAX_DIMS,
                "nbytes": len(b),      # <- variable
                "data_offset": 0,
                "reserved1": 0,
            })

        else:
            raise TypeError(f"Unsupported arg type in packet: {type(a)}")

    n_args = len(entries)

    header_size = 32
    entries_size = n_args * entry_size
    payload_offset = header_size + entries_size

    # Fill data offsets
    cur = payload_offset
    for e in entries:
        e["data_offset"] = cur
        cur += e["nbytes"]

    # Write file
    with open(packet_path, "wb") as f:
        # header: magic(8) version(u32) n_args(u32) entry_size(u32) reserved(u32) payload_offset(u64)
        f.write(struct.pack("<8sIIIIQ", MAGIC, VERSION, n_args, entry_size, 0, payload_offset))

        # entries
        for e in entries:
            f.write(ENTRY_STRUCT.pack(
                e["kind"],
                e["scalar_type"],
                e["tensor_dtype"],
                e["ndim"],
                e["flags"],
                e["reserved0"],
                *e["sizes"],
                *e["strides"],
                e["nbytes"],
                e["data_offset"],
                e["reserved1"],
            ))

        # payload
        for chunk in payload_chunks:
            f.write(chunk)
            
def _read_packet_bin_to_pyobjs(packet_path, device="cpu"):
    import struct
    import torch

    MAGIC_EXPECT = b"TTCIPKT1"
    MAX_DIMS = PACKET_MAX_DIMS

    ENTRY_STRUCT = struct.Struct(
        "<" +
        "6I" +
        f"{MAX_DIMS}Q" +
        f"{MAX_DIMS}q" +
        "3Q"
    )

    ENUM_TO_TORCH_DTYPE = {
        1: torch.float32,
        2: torch.float16,
        3: torch.bfloat16,
        4: torch.float64,
        5: torch.int32,
        6: torch.int64,
        7: torch.int16,
        8: torch.int8,
        9: torch.uint8,
        10: torch.bool,
    }

    KIND_SCALAR = 0
    KIND_TENSOR = 1

    SCALAR_INT64 = 1
    SCALAR_FLOAT64 = 2
    SCALAR_BOOL = 3
    SCALAR_BYTES = 4


    with open(packet_path, "rb") as f:
        header = f.read(32)
        if len(header) != 32:
            raise ValueError("packet.bin too small")

        magic, version, n_args, entry_size, _reserved, payload_offset = struct.unpack(
            "<8sIIIIQ", header
        )
        if magic != MAGIC_EXPECT:
            raise ValueError(f"Bad packet magic: {magic!r}")
        if version != 1:
            raise ValueError(f"Unsupported packet version: {version}")
        if entry_size != ENTRY_STRUCT.size:
            raise ValueError(f"Entry size mismatch: {entry_size} vs {ENTRY_STRUCT.size}")

        entries = []
        for i in range(n_args):
            eb = f.read(entry_size)
            if len(eb) != entry_size:
                raise ValueError(f"Truncated entry {i}")
            u = ENTRY_STRUCT.unpack(eb)
            kind = u[0]
            scalar_type = u[1]
            tensor_dtype = u[2]
            ndim = u[3]
            flags = u[4]
            reserved0 = u[5]
            sizes = list(u[6:6+MAX_DIMS])
            strides = list(u[6+MAX_DIMS:6+2*MAX_DIMS])
            nbytes = u[6+2*MAX_DIMS]
            data_offset = u[7+2*MAX_DIMS]
            reserved1 = u[8+2*MAX_DIMS]
            entries.append((kind, scalar_type, tensor_dtype, ndim, flags, reserved0, sizes, strides, nbytes, data_offset, reserved1))

        out = []
        for i, (kind, scalar_type, tensor_dtype, ndim, flags, reserved0, sizes, strides, nbytes, data_offset, reserved1) in enumerate(entries):
            f.seek(data_offset)
            payload = f.read(nbytes)
            if len(payload) != nbytes:
                raise ValueError(f"Truncated payload for arg {i}")

            if kind == KIND_SCALAR:
                if scalar_type != SCALAR_BYTES and nbytes != 8:
                    raise ValueError(f"Scalar must be 8 bytes (arg {i})")
                if scalar_type == SCALAR_INT64:
                    (v,) = struct.unpack("<q", payload)
                    out.append(v)
                elif scalar_type == SCALAR_FLOAT64:
                    (v,) = struct.unpack("<d", payload)
                    out.append(v)
                elif scalar_type == SCALAR_BOOL:
                    (v,) = struct.unpack("<Q", payload)
                    out.append(bool(v))
                elif scalar_type == SCALAR_BYTES:
                    # decode as utf-8 string; if you prefer raw bytes, return payload
                    out.append(payload.decode("utf-8"))
                else:
                    raise ValueError(f"Unknown scalar_type={scalar_type} (arg {i})")

            elif kind == KIND_TENSOR:
                dtype = ENUM_TO_TORCH_DTYPE.get(tensor_dtype)
                if dtype is None:
                    raise ValueError(f"Unknown tensor_dtype enum={tensor_dtype} (arg {i})")
                if ndim > MAX_DIMS:
                    raise ValueError(f"ndim too large: {ndim} (arg {i})")
                shape = tuple(int(x) for x in sizes[:ndim])
                logical_strides = tuple(int(x) for x in strides[:ndim])
                storage_offset = (int(reserved0) << 32) | int(flags)

                try:
                    elem_size = torch.empty((), dtype=dtype).element_size()
                    if nbytes == 0:
                        t = torch.empty(shape, dtype=dtype)
                    else:
                        backing = torch.frombuffer(bytearray(payload), dtype=dtype, count=nbytes // elem_size)
                        t = torch.as_strided(
                            backing,
                            size=shape,
                            stride=logical_strides,
                            storage_offset=storage_offset,
                        ).clone()
                except Exception:
                    # Very rare edge cases; fall back to byte tensor
                    t = torch.tensor(list(payload), dtype=torch.uint8).clone()
                    # Can't reshape meaningfully here; leave as flat uint8
                if device != "cpu":
                    t = t.to(device)
                out.append(t)
            else:
                raise ValueError(f"Unknown kind={kind} (arg {i})")

    return out


def _copy_back_packet_to_args(packet_path, args):
    """
    Read packet.bin and copy back values into the original args as much as possible.

    - For torch.Tensor args: in-place copy_ from packet tensor (shape/dtype must match).
    - For scalar args: returns a new args tuple with updated scalars.
    - If an arg is a tensor in args but packet provides scalar (or vice versa): raises.
    """
    import torch

    def _tensor_values_equal(orig, newv):
        try:
            lhs = orig.detach()
            if lhs.device.type != "cpu":
                lhs = lhs.cpu()
            rhs = newv.detach()
            lhs_bytes = lhs.contiguous().view(torch.uint8)
            rhs_bytes = rhs.contiguous().view(torch.uint8)
            return bool(torch.equal(lhs_bytes, rhs_bytes))
        except Exception:
            return False

    updated = _read_packet_bin_to_pyobjs(packet_path, device="cpu")
    if len(updated) != len(args):
        raise ValueError(f"Arg count mismatch: packet has {len(updated)}, args has {len(args)}")

    new_args = list(args)

    for i, (orig, newv) in enumerate(zip(args, updated)):
        # Tensor -> Tensor: copy back into original tensor storage
        if isinstance(orig, torch.Tensor):
            if not isinstance(newv, torch.Tensor):
                raise TypeError(f"Arg {i}: original is Tensor but packet is {type(newv)}")

            # enforce same dtype/shape for safe copy-back
            if orig.dtype != newv.dtype:
                raise ValueError(f"Arg {i}: dtype mismatch orig={orig.dtype} packet={newv.dtype}")
            if tuple(orig.shape) != tuple(newv.shape):
                raise ValueError(f"Arg {i}: shape mismatch orig={tuple(orig.shape)} packet={tuple(newv.shape)}")

            # Avoid writing back read-only inputs. This also sidesteps expanded
            # stride-0 views where copy_ is illegal even when the payload is
            # unchanged.
            if not _tensor_values_equal(orig, newv):
                orig.copy_(newv)

            # Keep original tensor object in args (no replacement)
            new_args[i] = orig

        else:
            # Scalar path: replace in returned args tuple
            if isinstance(newv, torch.Tensor):
                raise TypeError(f"Arg {i}: original is scalar-like but packet is Tensor")
            new_args[i] = newv
    
    return tuple(new_args)


def _generate_launcher(constants, signature, kernel_name, perf_matmul_enabled=False):
    arg_decls = ""
    for i, ty in signature.items():
        if ty == "constexpr":
            continue
        decl = ""
        if ty[0] == "*":        # tensor case
            decl = f"  GEN_MEMREF_ARG({i}, {_tensor_elem_ty_to_cpp(ty[1:])}, pkt)"
        else:                   # scalar case
            getter_expr = _scalar_packet_get_expr(
                ty, f"pkt.entries[{i}]", f"pkt.buffers[{i}]"
            )
            decl = f"  {_ty_to_cpp(ty)} arg{i} = {getter_expr};"
        arg_decls += decl + "\n"

    kernel_arg_decls = ', '.join(_ty_to_cpp(ty) if ty[0] != "*" else f"int64_t, void*" for i, ty in signature.items() if ty != "constexpr")
    kernel_arg_decls += ', ' if kernel_arg_decls else ''

    kernel_parameters = ', '.join(f"arg{i}" if ty[0] != "*" else f"arg{i}.rank, arg{i}.descriptor" for i, ty in signature.items() if ty != "constexpr")
    kernel_parameters += ', ' if kernel_parameters else ''

    perf_kernel_arg_decls = ""
    perf_kernel_parameters = ""
    perf_runtime_setup = ""
    perf_runtime_finalize = ""

    if perf_matmul_enabled:
        perf_kernel_arg_decls = "void*, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, "
        perf_kernel_parameters = (
            "matmulCyDesc.basePtr, matmulCyDesc.data, matmulCyDesc.offset, "
            "matmulCyDesc.sizes[0], matmulCyDesc.sizes[1], matmulCyDesc.sizes[2], "
            "matmulCyDesc.strides[0], matmulCyDesc.strides[1], matmulCyDesc.strides[2], "
        )
        perf_runtime_setup = """
  const int64_t gx = static_cast<int64_t>(gridX);
  const int64_t gy = static_cast<int64_t>(gridY);
  const int64_t gz = static_cast<int64_t>(gridZ);
  std::vector<int64_t> matmulCyStorage(static_cast<size_t>(gx * gy * gz), 0);
  StridedMemRefType<int64_t, 3> matmulCyDesc{};
  matmulCyDesc.basePtr = matmulCyStorage.data();
  matmulCyDesc.data = matmulCyStorage.data();
  matmulCyDesc.offset = 0;
  matmulCyDesc.sizes[0] = gx;
  matmulCyDesc.sizes[1] = gy;
  matmulCyDesc.sizes[2] = gz;
  matmulCyDesc.strides[0] = gy * gz;
  matmulCyDesc.strides[1] = gz;
  matmulCyDesc.strides[2] = 1;
"""
        perf_runtime_finalize = """
  int64_t matmulTotalCycles = 0;
  for (int32_t x = 0; x < gridX; ++x) {
    for (int32_t y = 0; y < gridY; ++y) {
      for (int32_t z = 0; z < gridZ; ++z) {
        const int64_t idx = (static_cast<int64_t>(x) * matmulCyDesc.strides[0]) +
                            (static_cast<int64_t>(y) * matmulCyDesc.strides[1]) +
                            static_cast<int64_t>(z);
        const int64_t v = matmulCyStorage[static_cast<size_t>(idx)];
        matmulTotalCycles += v;
      }
    }
  }
  if (matmulTotalCycles < 0) {
    matmulTotalCycles = 0;
  }
  if (pkt.entries.size() > 1) {
    pkt.entries[1].reserved1 = static_cast<uint64_t>(matmulTotalCycles);
  }
"""

    return f"""
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
static constexpr int MAX_DIMS = {PACKET_MAX_DIMS};

struct PacketHeader {{
  char     magic[8];
  uint32_t version;
  uint32_t n_args;
  uint32_t entry_size;
  uint32_t reserved;
  uint64_t payload_offset;
}};

struct PacketEntry {{
  uint32_t kind;
  uint32_t scalar_type;
  uint32_t tensor_dtype;
  uint32_t ndim;
  uint32_t flags;
  uint32_t reserved0;

  uint64_t sizes[MAX_DIMS];
  int64_t  strides[MAX_DIMS];

  uint64_t nbytes;
  uint64_t data_offset;
  uint64_t reserved1;
}};

struct PacketLoaded {{
  PacketHeader hdr{{}};
  std::vector<PacketEntry> entries;
  std::vector<void*> buffers;
  ~PacketLoaded();
}};

PacketLoaded load_packet(const char* packet_path);
void writeback_packet(const char* packet_path, const PacketLoaded& pkt);

// declare helper fns in driver.cpp
/*-------------scalar--------------*/
int64_t packet_get_i64(const PacketEntry& e, void* payload);
int32_t packet_get_i32(const PacketEntry& e, void* payload);
float packet_get_f32(const PacketEntry& e, void* payload);
double packet_get_f64(const PacketEntry& e, void* payload);
bool packet_get_bool(const PacketEntry& e, void* payload);
/*-------------tensor--------------*/
#define GEN_MEMREF_ARG(n, T, PKT)                                           \\
auto desc##n = StridedMemRefType<T, MAX_DIMS>{{}};                          \\
desc##n.basePtr = reinterpret_cast<T*>(PKT.buffers[n]);                     \\
desc##n.data    = reinterpret_cast<T*>(PKT.buffers[n]) +                    \\
                 static_cast<int64_t>(                                        \\
                     (static_cast<uint64_t>(PKT.entries[n].reserved0) << 32) | \\
                     static_cast<uint64_t>(PKT.entries[n].flags));           \\
desc##n.offset  = 0;                                                        \\
for(int i = 0; i < PKT.entries[n].ndim; i++){{                              \\
  desc##n.sizes[i] = static_cast<int64_t>(PKT.entries[n].sizes[i]);         \\
  desc##n.strides[i] = (PKT.entries[n].strides[i] != 0) ?                   \\
                      static_cast<int64_t>(PKT.entries[n].strides[i]) : 1;  \\
}}                                                                          \\
UnrankedMemRefType<T> arg##n{{PKT.entries[n].ndim, &desc##n}};

// Signature mirrors the lowered matmul_kernel in log-ttshared.o
extern "C" void {kernel_name}({kernel_arg_decls}{perf_kernel_arg_decls}
                              int32_t, int32_t, int32_t, int32_t, int32_t, int32_t); // gridDims, pids

static inline uint64_t rdcycle(void) {{
//  uint64_t x;
//  asm volatile ("rdcycle %0" : "=r"(x));
  return 0;
}}
                              
int main(int argc, char **argv) {{
  int32_t gridX = atoi(argv[1]);
  int32_t gridY = atoi(argv[2]);
  int32_t gridZ = atoi(argv[3]);
  char *packet_path = argv[4];

  PacketLoaded pkt = load_packet(packet_path);

  /*---------- declare args -------------*/
  {arg_decls}
  {perf_runtime_setup}
  /*----------- launch kernel -----------*/
  uint64_t t0 = rdcycle();
  for(int32_t x = 0; x < gridX; x++){{
    for(int32_t y = 0; y < gridY; y++){{
      for(int32_t z = 0; z < gridZ; z++){{
        {kernel_name}({kernel_parameters}{perf_kernel_parameters}
                        gridX, gridY, gridZ, x, y, z);
      }}
    }}
  }}
  uint64_t t1 = rdcycle();
  {perf_runtime_finalize}
  if (!pkt.entries.empty()) {{
    pkt.entries[0].reserved1 = static_cast<uint64_t>(t1 - t0);
  }}
  /*-------------write back--------------*/
  writeback_packet(packet_path, pkt);

  return 0;
}}
"""

def compile_module(launcher_src, kernel_placeholder_name, signature,
                   arg_names=None, perf_matmul_enabled=False):
    LLVM_PROJECT_PATH = _get_llvm_project_path()
    mlir_include_dir = os.path.join(LLVM_PROJECT_PATH, "mlir", "include")
    mlir_lib_dir = os.path.join(LLVM_PROJECT_PATH, "mlir", "lib")
    llvm_include_dir = os.path.join(LLVM_PROJECT_PATH, "llvm", "include")
    llvm_lib_dir = os.path.join(LLVM_PROJECT_PATH, "llvm", "lib")

    def launch(
        gridX, gridY, gridZ, stream, cu_function,
        kernel_metadata, launch_metadata,
        launch_enter_hook, launch_exit_hook, *args):
        
        # verilator simulation path is slow, so we skip pytorch-chipyard use of triton-chipayrd:
        # This path is used for single triton-chipyard kernel launch.
        sims_verilator = os.getenv("CHIPYARD_SIM_VERILATOR_PATH", "")
        if not sims_verilator:
            return True
        CHIPYARD_ENV_PATH = _get_chipyard_env_path()

        kernel_obj = cu_function
        kernel_name = kernel_metadata[6] # see pack_metadata in compiler.py
        custom_library_path = (
            str(kernel_metadata[10]) if len(kernel_metadata) > 10 else ""
        )
        custom_library_sha256 = (
            str(kernel_metadata[11]) if len(kernel_metadata) > 11 else ""
        )
        custom_library_bytes = b""
        if custom_library_path:
            custom_library = Path(custom_library_path)
            if not custom_library.is_file():
                raise FileNotFoundError(
                    f"custom Linalg archive was not found: {custom_library}"
                )
            custom_library_bytes = custom_library.read_bytes()
            actual_library_sha256 = hashlib.sha256(custom_library_bytes).hexdigest()
            if (
                custom_library_sha256
                and actual_library_sha256 != custom_library_sha256
            ):
                raise RuntimeError(
                    "custom Linalg archive changed after Triton compilation: "
                    f"{custom_library}"
                )
        src = launcher_src.replace(kernel_placeholder_name, kernel_name)
        driver_path = os.path.join(dirname, "driver.cpp")
        driver_src = Path(driver_path).read_bytes()
        key = hashlib.sha256(
            src.encode("utf-8")
            + kernel_obj
            + driver_src
            + custom_library_sha256.encode("utf-8")
            + custom_library_bytes
        ).hexdigest()
        cache = get_cache_manager(key)
        name = "__triton_chipyard_kernel_launcher"
        filename = f"{name}.out"
        cache_path = cache.get_file(filename)

        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                obj_path = os.path.join(tmpdir, "log-ttshared.o")
                launcher_src_path = os.path.join(tmpdir, "host.cpp")
                so_path = os.path.join(tmpdir, "a.out")
                march = os.getenv("TRITON_CHIPYARD_RISCV_MARCH", "").strip()
                mabi = os.getenv("TRITON_CHIPYARD_RISCV_MABI", "").strip()
                march_flag = f'-march="{march}"' if march else ""
                mabi_flag = f'-mabi="{mabi}"' if mabi else ""
                Path(obj_path).write_bytes(kernel_obj)
                Path(launcher_src_path).write_text(src)
                custom_library_arg = (
                    f'"{custom_library_path}"' if custom_library_path else ""
                )

                # gen a.out 
                cmd = f"""
source "$(conda info --base)/etc/profile.d/conda.sh" 
source "{CHIPYARD_ENV_PATH}" 
riscv64-unknown-linux-gnu-g++ \
{march_flag} {mabi_flag} \
"{launcher_src_path}" "{obj_path}" {custom_library_arg} "{driver_path}" \
"{mlir_lib_dir}"/ExecutionEngine/CRunnerUtils.cpp \
"{mlir_lib_dir}"/ExecutionEngine/RunnerUtils.cpp \
"{mlir_lib_dir}"/ExecutionEngine/Float16bits.cpp \
-lm \
-I"{mlir_include_dir}" -I"{llvm_include_dir}" \
-O2 -std=c++17 -static -o "{so_path}"
"""
                subprocess.check_call(["bash", "-lc", cmd])
                with open(so_path, "rb") as f:
                    cache_path = cache.put(f.read(), filename, binary=True)
        
        # Write torch tensors to packet file; prefer tmpfs to reduce I/O overhead.
        with tempfile.TemporaryDirectory(dir=_get_packet_tmpdir()) as tmpdir:
            packet_path = os.path.join(tmpdir, "packet.bin")
            _write_packet_bin(args, packet_path)

            # launch the kernel using verilator simulator
            cmd = f"""
source "$(conda info --base)/etc/profile.d/conda.sh"
source "{CHIPYARD_ENV_PATH}"
"{sims_verilator}" pk "{cache_path}" \
"{gridX}" "{gridY}" "{gridZ}" "{packet_path}"
"""
            subprocess.check_call(["bash", "-lc", cmd])

            cycles, matmul_cycles = _read_packet_perf_counters(packet_path)
            if not perf_matmul_enabled:
                matmul_cycles = None
            launch.last_cycles = cycles
            launch.last_matmul_cycles = matmul_cycles
            global _LAST_KERNEL_CYCLES, _LAST_MATMUL_CYCLES
            _LAST_KERNEL_CYCLES = cycles
            _LAST_MATMUL_CYCLES = matmul_cycles
            _copy_back_packet_to_args(packet_path, args)

    return launch


class ChipyardLauncher(object):

    def __init__(self, src, metadata):
        kernel_placeholder_name = "KERNEL_NAME_PLACEHOLDER"

        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        arg_names = None
        if hasattr(src, "fn") and hasattr(src.fn, "arg_names"):
            arg_names = {i: name for i, name in enumerate(src.fn.arg_names)}
        perf_matmul_enabled = _is_perf_matmul_enabled()
        launcher_src = _generate_launcher(
            constants,
            signature,
            kernel_placeholder_name,
            perf_matmul_enabled,
        )
        # Later KERNEL_NAME_PLACEHOLDER will be used to assign the kernel name
        # in the following launch function.
        self.launch = compile_module(
            launcher_src,
            kernel_placeholder_name,
            signature,
            arg_names=arg_names,
            perf_matmul_enabled=perf_matmul_enabled,
        )

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class ChipyardUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(ChipyardUtils, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def get_last_kernel_cycles():
        return _LAST_KERNEL_CYCLES

    @staticmethod
    def get_last_matmul_cycles():
        return _LAST_MATMUL_CYCLES

    # Note:
    # nvidia and amd backends have their corresponding driver.c file that exposes
    # get_device_properties and load_binary using python bindings.
    # (see third_party/nvidia/backend/driver.c)
    # These methods are then used in compiler.py to initialize handles before running
    # the triton kernels.
    # Since we recompile the kernel every time (see compile_module above),
    # and the metadata generated by these functions aren't applicable to the cpu
    # backend, just define the same functions with dummy implementation.
    @staticmethod
    def get_device_properties(device):
        return {
          "max_shared_mem": 2 ** 20,
          "multiprocessor_count": None,
          "sm_clock_rate": None,
          "mem_clock_rate": None,
          "mem_bus_width": None
        }

    # Important note:
    # Since we cannot easy pass function pointers around, we pass along the
    # obj of the kernel so that compile_module above can recompile the
    # module every time.
    @staticmethod
    def load_binary(name, kernel_obj, shared, device):
        return (
          None,       # module
          kernel_obj, # function
          None,       # n_regs
          None,        # n_spills
          sys.maxsize, # n_max_threads
        )

class _ChipyardEvent:
    def __init__(self, enable_timing=False):
        '''Dummy event class for benchmark
        should be impl'd with gemmini counter'''
        self.enable_timing = enable_timing
        self._time = None

    def record(self):
        if self.enable_timing:
            import time
            self._time = time.perf_counter()

    def elapsed_time(self, other):
        if self._time is None or other._time is None:
            return 0.0
        return (other._time - self._time) * 1000.0  # ms



class _ChipyardDeviceInterface:
    Event = _ChipyardEvent
    def synchronize(self):
        # CPU backend: usually synchronous by nature
        return None

class ChipyardDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = ChipyardUtils()
        self.launcher_cls = ChipyardLauncher
        self.binary_ext = "obj"

    # CPU driver won't be automatically chosen unless explicitly set through
    # triton.runtime.driver.set_active(ChipyardDriver())
    @staticmethod
    def is_active():
        return False
      
    def get_device_interface(self):
        return _ChipyardDeviceInterface()

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch
        cache_size = 256 * 1024 * 1024
        # something more needed for scratchpad flush
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cpu')
    
    def clear_cache(self, cache):
        return cache.zero_()

    def get_device_capability(self):
        return ("chipyard", 0)

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        # CPU doesn't have a device to return. Return something.
        return "chipyard"

    def set_current_device(self, device):
        # CPU doesn't have a device to set
        assert device == "chipyard"
        return

    def get_current_target(self):
        return GPUTarget("chipyard", 0, 0)

    def get_active_torch_device(self):
        import torch
        return torch.device("chipyard")

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args
    
    def map_python_to_cpp_type(self, ty: str) -> str:
        return _ty_to_cpp(ty)
  

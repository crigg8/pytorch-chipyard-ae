#ifndef TRITON_PERF_MATMUL_CONVERSION_PASSES_H
#define TRITON_PERF_MATMUL_CONVERSION_PASSES_H

#include "triton-shared/Conversion/PerfMatmul/PerfMatmul.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/PerfMatmul/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif

#ifndef TRITON_CONVERSION_PERFMATMUL_PERFMATMUL_H
#define TRITON_CONVERSION_PERFMATMUL_PERFMATMUL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/PerfMatmul/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createPerfMatmulPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_PERFMATMUL_PERFMATMUL_H

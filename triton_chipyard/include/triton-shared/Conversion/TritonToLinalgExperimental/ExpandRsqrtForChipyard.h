//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONTOLINALG_EXPANDRSQRTFORCHIPYARD_H
#define TRITON_CONVERSION_TRITONTOLINALG_EXPANDRSQRTFORCHIPYARD_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createExpandRsqrtForChipyardPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOLINALG_EXPANDRSQRTFORCHIPYARD_H

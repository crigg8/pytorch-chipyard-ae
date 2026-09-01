#ifndef TRITON_LINALG_TO_FUNCTION_CALL_H
#define TRITON_LINALG_TO_FUNCTION_CALL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToFunctionCallPass();

} // namespace mlir::triton

#endif

#ifndef TRITON_LINALG_TO_FUNCTION_CALL_PASSES_H
#define TRITON_LINALG_TO_FUNCTION_CALL_PASSES_H

#include "triton-shared/Conversion/LinalgToFunctionCall/LinalgToFunctionCall.h"

namespace mlir::triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/LinalgToFunctionCall/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/LinalgToFunctionCall/Passes.h.inc"

} // namespace mlir::triton

#endif

#ifndef LINALG_TO_FUNCTION_CALL_PASSES
#define LINALG_TO_FUNCTION_CALL_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToFunctionCall
    : Pass<"linalg-to-function-call", "mlir::ModuleOp"> {
  let summary = "Replace one bufferized Linalg op kind with an external call";
  let constructor = "mlir::triton::createLinalgToFunctionCallPass()";
  let options = [
    Option<"targetOp", "target-op", "std::string", /*default=*/"\"\"",
           "Exact Linalg operation name, for example linalg.matmul">,
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "External C function symbol">,
    Option<"operandTypes", "operand-types", "std::string", /*default=*/"\"\"",
           "Semicolon-separated element:ranks:logical-rank:unit-policy:canonicalize constraints">,
  ];
}

#endif

#ifndef PERF_MATMUL_CONVERSION_PASSES
#define PERF_MATMUL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def PerfMatmul : Pass<"perf-matmul", "mlir::ModuleOp"> {
  let summary = "Instrument linalg.matmul with cycle counters";
}

#endif

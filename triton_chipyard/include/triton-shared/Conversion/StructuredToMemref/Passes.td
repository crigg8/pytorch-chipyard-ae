#ifndef STRUCTURED_TO_MEMREF_CONVERSION_PASSES
#define STRUCTURED_TO_MEMREF_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def StructuredToMemref : Pass<"structured-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert triton structured pointer ops to memref";
  let options = [
      Option<"enableAliasFirst", "enable-alias-first", "bool", /*default*/"true",
             "Alias eligible structured loads instead of always materializing an alloc+copy temporary">,
  ];
}

#endif

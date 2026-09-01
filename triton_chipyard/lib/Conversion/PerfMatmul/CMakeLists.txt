add_triton_library(PerfMatmul
  PerfMatmulPass.cpp

  DEPENDS
  PerfMatmulConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSupport
  TritonIR
  MLIRTensorDialect
  MLIRTransforms
  TritonTransforms
)

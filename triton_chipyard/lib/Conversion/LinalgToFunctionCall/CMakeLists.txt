add_triton_library(LinalgToFunctionCall
  LinalgToFunctionCallPass.cpp

  DEPENDS
  LinalgToFunctionCallConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRDestinationStyleOpInterface
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSupport
)

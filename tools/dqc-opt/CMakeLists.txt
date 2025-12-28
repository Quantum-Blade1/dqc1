add_llvm_executable(dqc-opt
  dqc-opt.cpp
)

target_link_libraries(dqc-opt
  PRIVATE
  DQCCompiler
  DQCDialect
  MLIRIR
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
)

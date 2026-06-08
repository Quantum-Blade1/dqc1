#===---------- CMakeLists.txt for lib/Lowering ----------===

add_llvm_library(
  DQCLoweringPasses
  MPILoweringPass.cpp
  LLVMLoweringPass.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${DQC_INCLUDE_DIRS}

  LINK_LIBS
  PUBLIC
  DQCDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRFuncToLLVM
)

target_include_directories(DQCLoweringPasses PUBLIC ${DQC_INCLUDE_DIRS})
add_dependencies(DQCLoweringPasses DQCIncGen)
llvm_update_compile_flags(DQCLoweringPasses)

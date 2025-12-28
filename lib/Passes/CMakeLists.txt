#===---------- CMakeLists.txt for lib/Passes ----------===

add_subdirectory(Partitioning)
add_subdirectory(Synthesis)
add_subdirectory(Optimization)

add_llvm_library(
  DQCPasses
  PassRegistry.cpp

  ADDITIONAL_HEADER_DIRS
  ${DQC_INCLUDE_DIRS}

  LINK_LIBS
  PUBLIC
  DQCDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  MLIRArithDialect
)

target_include_directories(DQCPasses PUBLIC ${DQC_INCLUDE_DIRS})
add_dependencies(DQCPasses DQCIncGen)
llvm_update_compile_flags(DQCPasses)

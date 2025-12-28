#===---------- CMakeLists.txt for lib/Passes/Optimization ----------===

add_llvm_library(
  DQCOptimizationPasses
  GreedyReorderingPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${DQC_INCLUDE_DIRS}
)

target_include_directories(DQCOptimizationPasses PUBLIC ${DQC_INCLUDE_DIRS})
add_dependencies(DQCOptimizationPasses DQCIncGen)
llvm_update_compile_flags(DQCOptimizationPasses)

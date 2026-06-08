#===---------- CMakeLists.txt for lib/Passes/Partitioning ----------===

add_llvm_library(
  DQCPartitioningPasses
  InteractionGraphPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${DQC_INCLUDE_DIRS}
)

target_include_directories(DQCPartitioningPasses PUBLIC ${DQC_INCLUDE_DIRS})
add_dependencies(DQCPartitioningPasses DQCIncGen)
llvm_update_compile_flags(DQCPartitioningPasses)

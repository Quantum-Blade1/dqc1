#===---------- CMakeLists.txt for lib ----------===

add_subdirectory(Dialect)
add_subdirectory(Passes)
add_subdirectory(Lowering)

add_llvm_library(
  DQCCompiler
  Init.cpp

  ADDITIONAL_HEADER_DIRS
  ${DQC_INCLUDE_DIRS}

  DEPENDS
  DQCIncGen
)

target_include_directories(DQCCompiler PUBLIC ${DQC_INCLUDE_DIRS})

# Link against dialect and passes
target_link_libraries(DQCCompiler
  PUBLIC
  DQCDialect
  DQCPasses
  DQCPartitioningPasses
  DQCSynthesisPasses
  DQCOptimizationPasses
  DQCLoweringPasses
  MLIRIR
  MLIRTransforms
  MLIRSupport
)

add_dependencies(DQCCompiler DQCIncGen)

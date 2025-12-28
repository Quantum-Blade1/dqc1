#===---------- CMakeLists.txt for lib/Passes/Synthesis ----------===

add_llvm_library(
  DQCSynthesisPasses
  TeleGateSynthesisPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${DQC_INCLUDE_DIRS}
)

target_include_directories(DQCSynthesisPasses PUBLIC ${DQC_INCLUDE_DIRS})
add_dependencies(DQCSynthesisPasses DQCIncGen DQCDialect)
llvm_update_compile_flags(DQCSynthesisPasses)

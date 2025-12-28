#===---------- DQC Dialect TableGen ----------===

set(LLVM_TARGET_DEFINITIONS DQCOps.td)
mlir_tablegen(DQCOps.h.inc -gen-op-decls)
mlir_tablegen(DQCOps.cpp.inc -gen-op-defs)
mlir_tablegen(DQCDialect.h.inc -gen-dialect-decls)
mlir_tablegen(DQCDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(DQCIncGen)

add_llvm_library(
  DQCDialect
  DQCDialect.cpp
  DQCOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${DQC_INCLUDE_DIRS}
)

target_include_directories(DQCDialect PUBLIC ${DQC_INCLUDE_DIRS})
add_dependencies(DQCDialect DQCIncGen)

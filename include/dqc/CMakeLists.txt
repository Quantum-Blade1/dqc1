#===---------- DQC Include Directory ----------===

set(LLVM_TARGET_DEFINITIONS ${DQC_MAIN_INCLUDE_DIR}/dqc/DQCOps.td)
mlir_tablegen(${CMAKE_CURRENT_BINARY_DIR}/DQCOps.h.inc -gen-op-decls)
mlir_tablegen(${CMAKE_CURRENT_BINARY_DIR}/DQCOps.cpp.inc -gen-op-defs)
mlir_tablegen(${CMAKE_CURRENT_BINARY_DIR}/DQCDialect.h.inc -gen-dialect-decls)
mlir_tablegen(${CMAKE_CURRENT_BINARY_DIR}/DQCDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(DQCIncGen)

# Add a custom target to copy generated files to the include directory
add_custom_target(CopyDQCInc ALL
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/DQCOps.h.inc
    ${DQC_MAIN_INCLUDE_DIR}/dqc/DQCOps.h.inc
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/DQCOps.cpp.inc
    ${DQC_MAIN_INCLUDE_DIR}/dqc/DQCOps.cpp.inc
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/DQCDialect.h.inc
    ${DQC_MAIN_INCLUDE_DIR}/dqc/DQCDialect.h.inc
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/DQCDialect.cpp.inc
    ${DQC_MAIN_INCLUDE_DIR}/dqc/DQCDialect.cpp.inc
  DEPENDS DQCIncGen
)

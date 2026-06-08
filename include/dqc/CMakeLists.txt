#===---------- DQC Include Directory ----------===

set(LLVM_TARGET_DEFINITIONS ${DQC_MAIN_INCLUDE_DIR}/dqc/DQCDialect.td)
mlir_tablegen(DQCOps.h.inc -gen-op-decls)
mlir_tablegen(DQCOps.cpp.inc -gen-op-defs)
mlir_tablegen(DQCDialect.h.inc -gen-dialect-decls)
mlir_tablegen(DQCDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(DQCIncGen)

set(LLVM_TARGET_DEFINITIONS ${DQC_MAIN_INCLUDE_DIR}/dqc/MPIDialect.td)
mlir_tablegen(MPIOps.h.inc -gen-op-decls)
mlir_tablegen(MPIOps.cpp.inc -gen-op-defs)
mlir_tablegen(MPIDialect.h.inc -gen-dialect-decls)
mlir_tablegen(MPIDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MPIIncGen)

# Copy generated files to the include directory
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
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/MPIOps.h.inc
    ${DQC_MAIN_INCLUDE_DIR}/dqc/MPIOps.h.inc
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/MPIOps.cpp.inc
    ${DQC_MAIN_INCLUDE_DIR}/dqc/MPIOps.cpp.inc
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/MPIDialect.h.inc
    ${DQC_MAIN_INCLUDE_DIR}/dqc/MPIDialect.h.inc
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/MPIDialect.cpp.inc
    ${DQC_MAIN_INCLUDE_DIR}/dqc/MPIDialect.cpp.inc
  DEPENDS DQCIncGen MPIIncGen
)

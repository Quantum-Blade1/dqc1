#===---------- test/IR CMakeLists.txt ----------===

# IR validation tests
add_test(NAME DQCIRTests COMMAND dqc-opt ${CMAKE_CURRENT_SOURCE_DIR}/dqc_ir_test.mlir)

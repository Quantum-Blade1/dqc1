#===---------- test/Passes CMakeLists.txt ----------===

# Pass tests
add_mlir_pass_test(DQCPassTests
  interaction_graph_test.mlir
  telegate_synthesis_test.mlir
  reordering_test.mlir
)

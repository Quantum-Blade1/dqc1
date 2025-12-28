#===---------- test/Passes CMakeLists.txt ----------===

# Pass tests
add_test(NAME DQCPassTest_InteractionGraph COMMAND dqc-opt ${CMAKE_CURRENT_SOURCE_DIR}/interaction_graph_test.mlir)
add_test(NAME DQCPassTest_TeleGateSynthesis COMMAND dqc-opt ${CMAKE_CURRENT_SOURCE_DIR}/telegate_synthesis_test.mlir)
add_test(NAME DQCPassTest_Reordering COMMAND dqc-opt ${CMAKE_CURRENT_SOURCE_DIR}/reordering_test.mlir)

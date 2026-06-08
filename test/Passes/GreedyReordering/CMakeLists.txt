### CMake rules for GreedyReordering pass tests

find_program(DQC_OPT_EXECUTABLE dqc-opt)
find_program(FILECHECK_EXECUTABLE FileCheck)

if(DQC_OPT_EXECUTABLE AND FILECHECK_EXECUTABLE)
  # Basic transformation test
  add_test(NAME DQCPass_GreedyReorder_Basic
    COMMAND ${CMAKE_COMMAND} -E sh -c "${DQC_OPT_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/basic.mlir --dqc-greedy-reordering --ebit-reduction-target=0.3 2>&1 | ${FILECHECK_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/basic.mlir")

  # Edge / degenerate input
  add_test(NAME DQCPass_GreedyReorder_Edge
    COMMAND ${CMAKE_COMMAND} -E sh -c "${DQC_OPT_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/edge.mlir --dqc-greedy-reordering 2>&1 | ${FILECHECK_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/edge.mlir")

  # Invalid input should emit a diagnostic (check stderr)
  add_test(NAME DQCPass_GreedyReorder_Invalid
    COMMAND ${CMAKE_COMMAND} -E sh -c "${DQC_OPT_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/invalid.mlir --dqc-greedy-reordering 2>&1 | ${FILECHECK_EXECUTABLE} --check-prefix=ERR ${CMAKE_CURRENT_SOURCE_DIR}/invalid.mlir")
else()
  message(STATUS "Skipping GreedyReordering tests; dqc-opt or FileCheck not found")
endif()

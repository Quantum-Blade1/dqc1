#===---------- CMakeLists.txt for runtime ----------===
# DQC quantum simulator runtime library (plain C, no MLIR/LLVM deps)

add_library(dqc_runtime STATIC
  state_vector.c
  gates.c
  measurement.c
  mpi_comm.c
)

target_include_directories(dqc_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(dqc_runtime PROPERTIES
  C_STANDARD 11
  POSITION_INDEPENDENT_CODE ON
)

# Link math library
target_link_libraries(dqc_runtime PRIVATE m)

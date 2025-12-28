cmake_minimum_required(VERSION 3.13.4)
project(dqc-compiler LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

#===---------- MLIR Configuration ----------===

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

#===---------- DQC Compiler Configuration ----------===

set(DQC_MAIN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DQC_INCLUDE_DIRS "${DQC_MAIN_INCLUDE_DIR}")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)

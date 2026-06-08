// RUN: not dqc-opt %s --dqc-mpi-lowering 2>&1 | FileCheck %s --check-prefix=ERR
//===- invalid.mlir - MPI Lowering invalid input ---*- MLIR -*-===//

// Malformed IR should produce a diagnostic
module {
func.func @broken( {
  // ERR: error:
}

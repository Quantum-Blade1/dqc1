// RUN: dqc-opt %s --dqc-mpi-lowering | FileCheck %s
//===- edge.mlir - MPI Lowering edge cases ---*- MLIR -*-===//

// Empty module should be handled without errors
module {
func.func @empty() {
  // CHECK: func.func @empty()
  return
}
}

// Single-op that is unrelated should be preserved
func.func @single_op(%q: !dqc.qubit) {
  // CHECK-LABEL: func.func @single_op(
  dqc.local_gate %q : (!dqc.qubit)
  // CHECK: dqc.local_gate
  return
}

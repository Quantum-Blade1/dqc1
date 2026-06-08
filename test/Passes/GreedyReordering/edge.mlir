// RUN: dqc-opt %s --dqc-greedy-reordering | FileCheck %s
//===- edge.mlir - GreedyReordering edge / degenerate cases ---*- MLIR -*-===//

// Minimal function: ensure the pass is a no-op and deterministic on empty bodies
func.func @empty_function() {
  // CHECK: func.func @empty_function()
  return
}

// Single-op function: ensure single-op handling
func.func @single_op(%q0: !dqc.qubit) {
  // CHECK-LABEL: func.func @single_op(
  dqc.local_gate %q0 : (!dqc.qubit)
  // CHECK: dqc.local_gate
  return
}

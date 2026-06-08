// RUN: dqc-opt %s --dqc-interaction-graph | FileCheck %s
//===- edge.mlir - Partitioning edge cases ---*- MLIR -*-===//

// Empty function
func.func @empty() {
  // CHECK: func.func @empty()
  return
}

// Function with no cnot-like ops should still get partition metadata
func.func @no_interactions() {
  // CHECK: func.func @no_interactions()
  // CHECK: dqc.partition
  // CHECK: dqc.edge_cut_cost
  return
}

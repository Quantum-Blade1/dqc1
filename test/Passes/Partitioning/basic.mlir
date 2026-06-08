// RUN: dqc-opt %s --dqc-interaction-graph | FileCheck %s
//===- basic.mlir - Partitioning basic case ---*- MLIR -*-===//

func.func @partition_simple() {
  // CHECK: func.func @partition_simple()
  // CHECK-SAME: dqc.partition
  // CHECK-SAME: dqc.edge_cut_cost

  %q0 = dqc.alloc_qubit : !dqc.qubit
  %q1 = dqc.alloc_qubit : !dqc.qubit
  dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)

  return
}

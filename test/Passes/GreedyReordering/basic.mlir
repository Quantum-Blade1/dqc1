// RUN: dqc-opt %s --dqc-greedy-reordering --ebit-reduction-target=0.3 | FileCheck %s
//===- basic.mlir - GreedyReordering basic case ---*- MLIR -*-===//

func.func @gate_packet_optimization() {
  // CHECK: func.func @gate_packet_optimization()

  // Initial layout: two telegate operations sharing the same control qubit
  %q0 = dqc.alloc_qubit : !dqc.qubit
  %q1 = dqc.alloc_qubit : !dqc.qubit

  // These two distributed gates should be grouped by the pass
  dqc.telegate %q0, %q1 : (!dqc.qubit, !dqc.qubit)
  dqc.telegate %q0, %q1 : (!dqc.qubit, !dqc.qubit)

  // A local gate that should remain local
  dqc.local_gate %q1 : (!dqc.qubit)

  // CHECK-NEXT: dqc.telegate
  // CHECK-NEXT: dqc.telegate

  return
}

// RUN: dqc-opt %s --dqc-telegate-synthesis | FileCheck %s
//===- edge.mlir - TeleGate Synthesis edge cases ---*- MLIR -*-===//

func.func @empty() {
  // CHECK: func.func @empty()
  return
}

func.func @local_only(%q: !dqc.qubit) {
  // CHECK-LABEL: func.func @local_only(
  dqc.local_gate %q : (!dqc.qubit)
  // CHECK: dqc.local_gate
  return
}

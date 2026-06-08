// RUN: dqc-opt %s --dqc-telegate-synthesis | FileCheck %s
//===- basic.mlir - TeleGate Synthesis basic case ---*- MLIR -*-===//

func.func @synthesis_simple() {
  // CHECK: func.func @synthesis_simple()

  // cnot between qubits assigned to different QPUs should be replaced
  %q0 = dqc.alloc_qubit : !dqc.qubit
  %q1 = dqc.alloc_qubit : !dqc.qubit
  dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)

  // CHECK: dqc.epr_alloc
  // CHECK: dqc.telegate

  return
}

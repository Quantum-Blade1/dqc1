//===- telegate_synthesis_test.mlir - Phase B Tests ---*- MLIR -*-===//
//
// Tests for the TeleGate Synthesis Pass (Phase B)
// Tests conversion of inter-QPU CNOTs to TeleGate sequences
//
//===------------------------------------------------------------===//

// RUN: mlir-opt %s --dqc-telegate-synthesis | FileCheck %s

func.func @distributed_cnot(%q0 : !quir.qubit, %q1 : !quir.qubit) -> !quir.qubit {
  // CHECK: dqc.epr_alloc
  // CHECK: dqc.telegate
  %result = quir.cnot %q0, %q1 : !quir.qubit, !quir.qubit -> !quir.qubit
  return %result : !quir.qubit
}

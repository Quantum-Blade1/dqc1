//===- interaction_graph_test.mlir - Phase A Tests ---*- MLIR -*-===//
//
// Tests for the Interaction Graph Pass (Phase A)
// Tests qubit-to-hypergraph mapping
//
//===---------------------------------------------------------------===//

// RUN: mlir-opt %s --dqc-interaction-graph -output-graph=/tmp/test_hypergraph.hmetis | FileCheck %s

func.func @simple_circuit(%q0 : !quir.qubit, %q1 : !quir.qubit) -> !quir.qubit {
  // CHECK: dqc.partition_info
  %result = quir.cnot %q0, %q1 : !quir.qubit, !quir.qubit -> !quir.qubit
  return %result : !quir.qubit
}

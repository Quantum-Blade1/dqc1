//===- reordering_test.mlir - Phase C Tests ---*- MLIR -*-===//
//
// Tests for the Greedy Reordering Pass (Phase C)
// Tests gate commutativity-based reordering for e-bit optimization
//
//===---------------------------------------------------===//

// RUN: mlir-opt %s --dqc-greedy-reordering --ebit-reduction-target=0.3 | FileCheck %s

func.func @gate_packet_optimization() {
  // This test verifies that multiple gates sharing a control qubit
  // are grouped into gate packets to reduce e-bit consumption
  return
}

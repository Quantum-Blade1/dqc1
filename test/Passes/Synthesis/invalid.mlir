// RUN: not dqc-opt %s --dqc-telegate-synthesis 2>&1 | FileCheck %s --check-prefix=ERR
//===- invalid.mlir - TeleGate Synthesis invalid input ---*- MLIR -*-===//

func.func @broken( {
  // ERR: error:
}

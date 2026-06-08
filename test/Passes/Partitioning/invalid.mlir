// RUN: not dqc-opt %s --dqc-interaction-graph 2>&1 | FileCheck %s --check-prefix=ERR
//===- invalid.mlir - Partitioning invalid input ---*- MLIR -*-===//

func.func @broken( {
  // ERR: error:
}

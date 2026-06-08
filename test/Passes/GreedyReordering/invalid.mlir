// RUN: dqc-opt %s --dqc-greedy-reordering 2>&1 | FileCheck %s --check-prefix=ERR
//===- invalid.mlir - GreedyReordering invalid input tests ---*- MLIR -*-===//

// This file intentionally contains malformed IR to assert that the pass
// and toolchain emit clear diagnostics instead of crashing.

func.func @broken( {
  // ERR: error:
}

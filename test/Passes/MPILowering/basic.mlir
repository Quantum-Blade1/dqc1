// RUN: dqc-opt %s --dqc-mpi-lowering | FileCheck %s
//===- basic.mlir - MPI Lowering basic case ---*- MLIR -*-===//

module {
func.func @mpi_lowering_simple() {
  // CHECK: func.func @mpi_lowering_simple()

  // EPR allocation should be lowered to mpi.distribute_epr
  %epr = dqc.epr_alloc 0, 1 : !dqc.epr_handle
  // A teleported gate
  dqc.telegate %0, %1, %epr : (!dqc.qubit, !dqc.qubit, !dqc.epr_handle)

  // CHECK: mpi.distribute_epr
  // CHECK: mpi.telegate_sequence

  return
}
}

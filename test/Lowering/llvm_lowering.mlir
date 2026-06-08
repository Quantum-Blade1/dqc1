// RUN: dqc-opt %s --dqc-llvm-lowering | FileCheck %s

module {
  func.func @test_lowering() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit

    dqc.h %q0 : (!dqc.qubit)
    dqc.x %q1 : (!dqc.qubit)
    dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)
    dqc.rx %q0 1.5707963 : (!dqc.qubit)
    %c = dqc.measure %q0 : (!dqc.qubit) -> !dqc.cbit

    return
  }
}

// CHECK: llvm.func @test_lowering
// CHECK: llvm.call @dqc_init
// CHECK: llvm.call @dqc_alloc_qubit
// CHECK: llvm.call @dqc_alloc_qubit
// CHECK: llvm.call @dqc_h
// CHECK: llvm.call @dqc_x
// CHECK: llvm.call @dqc_cnot
// CHECK: llvm.call @dqc_rx
// CHECK: llvm.call @dqc_measure
// CHECK: llvm.call @dqc_dump_state
// CHECK: llvm.call @dqc_finalize
// CHECK: llvm.return

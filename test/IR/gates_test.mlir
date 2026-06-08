// RUN: dqc-opt %s -o /dev/null
//===- gates_test.mlir - Test all DQC gate operations ---*- MLIR -*-===//

func.func @test_single_qubit_gates() {
  %q = dqc.alloc_qubit : !dqc.qubit

  // Standard gates
  dqc.h %q : (!dqc.qubit)
  dqc.x %q : (!dqc.qubit)
  dqc.y %q : (!dqc.qubit)
  dqc.z %q : (!dqc.qubit)
  dqc.s %q : (!dqc.qubit)
  dqc.t %q : (!dqc.qubit)

  // Parametric rotations
  dqc.rx %q 1.5707963 : (!dqc.qubit)
  dqc.ry %q 3.1415926 : (!dqc.qubit)
  dqc.rz %q 0.7853981 : (!dqc.qubit)

  // Legacy local_gate still works
  dqc.local_gate %q : (!dqc.qubit)

  return
}

func.func @test_two_qubit_gates() {
  %q0 = dqc.alloc_qubit : !dqc.qubit
  %q1 = dqc.alloc_qubit : !dqc.qubit

  dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)
  dqc.cz %q0, %q1 : (!dqc.qubit, !dqc.qubit)
  dqc.swap %q0, %q1 : (!dqc.qubit, !dqc.qubit)

  return
}

func.func @test_three_qubit_gate() {
  %q0 = dqc.alloc_qubit : !dqc.qubit
  %q1 = dqc.alloc_qubit : !dqc.qubit
  %q2 = dqc.alloc_qubit : !dqc.qubit

  dqc.ccx %q0, %q1, %q2 : (!dqc.qubit, !dqc.qubit, !dqc.qubit)

  return
}

func.func @test_measurement() {
  %q = dqc.alloc_qubit : !dqc.qubit
  dqc.h %q : (!dqc.qubit)
  %c = dqc.measure %q : (!dqc.qubit) -> !dqc.cbit

  return
}

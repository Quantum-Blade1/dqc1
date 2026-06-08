// Example 6: Gate Showcase — Every Gate in the System
//
// Demonstrates ALL available quantum gates in the DQC compiler
// This is a reference for what you can use in your circuits

module {
  func.func @all_gates() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit
    %q2 = dqc.alloc_qubit : !dqc.qubit

    // ---- Single-Qubit Gates ----

    dqc.h %q0 : (!dqc.qubit)      // Hadamard — creates superposition
    dqc.x %q0 : (!dqc.qubit)      // Pauli-X — bit flip (like NOT)
    dqc.y %q0 : (!dqc.qubit)      // Pauli-Y — bit flip + phase
    dqc.z %q0 : (!dqc.qubit)      // Pauli-Z — phase flip
    dqc.s %q0 : (!dqc.qubit)      // S gate — pi/4 phase
    dqc.t %q0 : (!dqc.qubit)      // T gate — pi/8 phase

    // ---- Rotation Gates (take an angle in radians) ----

    dqc.rx %q1 1.5707963 : (!dqc.qubit)  // Rotate around X axis (pi/2)
    dqc.ry %q1 0.7853981 : (!dqc.qubit)  // Rotate around Y axis (pi/4)
    dqc.rz %q1 3.1415926 : (!dqc.qubit)  // Rotate around Z axis (pi)

    // ---- Two-Qubit Gates ----

    dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)  // Controlled-NOT
    dqc.cz %q0, %q1 : (!dqc.qubit, !dqc.qubit)    // Controlled-Z
    dqc.swap %q0, %q1 : (!dqc.qubit, !dqc.qubit)  // Swap two qubits

    // ---- Three-Qubit Gate ----

    dqc.ccx %q0, %q1, %q2 : (!dqc.qubit, !dqc.qubit, !dqc.qubit)  // Toffoli (AND gate)

    // ---- Measurement ----

    %c0 = dqc.measure %q0 : (!dqc.qubit) -> !dqc.cbit  // Measure → classical bit

    return
  }
}

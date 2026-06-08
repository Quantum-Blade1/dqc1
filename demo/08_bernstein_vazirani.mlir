// Example 8: Bernstein-Vazirani Algorithm
//
// A quantum algorithm that finds a hidden binary string in ONE query!
// Classically you'd need N queries (one per bit). Quantum: just ONE.
//
// Hidden string: s = 1101
// The algorithm will measure exactly 1101 every time.
//
// How it works:
//   1. Hadamard all query qubits
//   2. Oracle: CNOT from each qubit where s_i = 1 to ancilla
//   3. Hadamard all query qubits again
//   4. Measure → get the hidden string!

module {
  func.func @bernstein_vazirani() {
    // 4 query qubits + 1 ancilla
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit
    %q2 = dqc.alloc_qubit : !dqc.qubit
    %q3 = dqc.alloc_qubit : !dqc.qubit
    %anc = dqc.alloc_qubit : !dqc.qubit

    // Prepare ancilla in |-> state
    dqc.x %anc : (!dqc.qubit)
    dqc.h %anc : (!dqc.qubit)

    // Hadamard on all query qubits
    dqc.h %q0 : (!dqc.qubit)
    dqc.h %q1 : (!dqc.qubit)
    dqc.h %q2 : (!dqc.qubit)
    dqc.h %q3 : (!dqc.qubit)

    // Oracle for s = 1101
    dqc.cnot %q0, %anc : (!dqc.qubit, !dqc.qubit)  // s[0] = 1
    dqc.cnot %q1, %anc : (!dqc.qubit, !dqc.qubit)  // s[1] = 1
    // s[2] = 0 → no CNOT
    dqc.cnot %q3, %anc : (!dqc.qubit, !dqc.qubit)  // s[3] = 1

    // Hadamard again
    dqc.h %q0 : (!dqc.qubit)
    dqc.h %q1 : (!dqc.qubit)
    dqc.h %q2 : (!dqc.qubit)
    dqc.h %q3 : (!dqc.qubit)

    // Measure — should always give 1101
    %c0 = dqc.measure %q0 : (!dqc.qubit) -> !dqc.cbit
    %c1 = dqc.measure %q1 : (!dqc.qubit) -> !dqc.cbit
    %c2 = dqc.measure %q2 : (!dqc.qubit) -> !dqc.cbit
    %c3 = dqc.measure %q3 : (!dqc.qubit) -> !dqc.cbit

    return
  }
}

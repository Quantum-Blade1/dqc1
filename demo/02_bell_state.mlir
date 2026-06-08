// Example 2: Bell State — Quantum Entanglement
//
// Creates the famous Bell state: (|00> + |11>) / sqrt(2)
// Two qubits become entangled — measuring one instantly determines the other
//
// Expected output:
//   |00> : prob = 0.5000
//   |11> : prob = 0.5000
//   (no |01> or |10> — that's entanglement!)

module {
  func.func @bell_state() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit

    // Step 1: Put q0 in superposition
    dqc.h %q0 : (!dqc.qubit)

    // Step 2: Entangle q0 and q1 with CNOT
    dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)

    return
  }
}

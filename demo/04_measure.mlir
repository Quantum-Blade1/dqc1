// Example 4: Measurement — Quantum Randomness
//
// Creates a Bell state then measures both qubits
// Run this multiple times — you'll get DIFFERENT results each time!
// But q0 and q1 always match (both 0 or both 1)
//
// This demonstrates:
//   1. Quantum superposition (50/50 chance)
//   2. Entanglement (qubits are correlated)
//   3. Wavefunction collapse (measurement destroys superposition)

module {
  func.func @measure_demo() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit

    dqc.h %q0 : (!dqc.qubit)
    dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)

    // Measure collapses the quantum state
    %c0 = dqc.measure %q0 : (!dqc.qubit) -> !dqc.cbit
    %c1 = dqc.measure %q1 : (!dqc.qubit) -> !dqc.cbit

    return
  }
}

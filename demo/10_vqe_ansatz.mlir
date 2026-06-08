// Example 10: VQE Ansatz — Variational Quantum Eigensolver
//
// VQE is used for quantum chemistry — finding ground state energies
// of molecules. It's one of the most promising near-term quantum algorithms.
//
// This is a "hardware-efficient ansatz" with:
//   - Layers of Ry/Rz rotations (the variational parameters)
//   - CNOT entangling layers between them
//
// In real VQE, a classical optimizer tunes the angles.
// Here we use fixed angles to show the circuit structure.

module {
  func.func @vqe_ansatz() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit
    %q2 = dqc.alloc_qubit : !dqc.qubit
    %q3 = dqc.alloc_qubit : !dqc.qubit

    // Layer 1: Ry rotations
    dqc.ry %q0 0.5236 : (!dqc.qubit)
    dqc.ry %q1 1.0472 : (!dqc.qubit)
    dqc.ry %q2 1.5708 : (!dqc.qubit)
    dqc.ry %q3 2.0944 : (!dqc.qubit)

    // Entangling layer 1
    dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q1, %q2 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q2, %q3 : (!dqc.qubit, !dqc.qubit)

    // Layer 2: Rz + Ry rotations
    dqc.rz %q0 0.7854 : (!dqc.qubit)
    dqc.rz %q1 1.1781 : (!dqc.qubit)
    dqc.rz %q2 0.3927 : (!dqc.qubit)
    dqc.rz %q3 1.9635 : (!dqc.qubit)

    dqc.ry %q0 1.8326 : (!dqc.qubit)
    dqc.ry %q1 0.6109 : (!dqc.qubit)
    dqc.ry %q2 2.4435 : (!dqc.qubit)
    dqc.ry %q3 0.9163 : (!dqc.qubit)

    // Entangling layer 2 (circular)
    dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q1, %q2 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q2, %q3 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q3, %q0 : (!dqc.qubit, !dqc.qubit)

    return
  }
}

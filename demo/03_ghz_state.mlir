// Example 3: GHZ State — Multi-Qubit Entanglement
//
// Greenberger-Horne-Zeilinger state: (|0000> + |1111>) / sqrt(2)
// ALL 4 qubits entangled — they're either ALL 0 or ALL 1
//
// This is what makes quantum networking interesting:
// The compiler will distribute qubits across QPUs and use
// quantum teleportation for cross-QPU CNOT gates!
//
// Expected output:
//   |0000> : prob = 0.5000
//   |1111> : prob = 0.5000

module {
  func.func @ghz_4() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit
    %q2 = dqc.alloc_qubit : !dqc.qubit
    %q3 = dqc.alloc_qubit : !dqc.qubit

    dqc.h %q0 : (!dqc.qubit)
    dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q1, %q2 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q2, %q3 : (!dqc.qubit, !dqc.qubit)

    return
  }
}

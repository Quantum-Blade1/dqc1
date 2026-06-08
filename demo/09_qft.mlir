// Example 9: Quantum Fourier Transform (4 qubits)
//
// QFT is the quantum version of the Fast Fourier Transform
// It's the key building block of Shor's factoring algorithm
//
// Applied to |0000>, it produces a uniform superposition
// over all 16 basis states (each with probability 1/16)
//
// The compiler distributes this across 2 QPUs and uses
// quantum teleportation for the cross-QPU CNOT gates

module {
  func.func @qft_4() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit
    %q2 = dqc.alloc_qubit : !dqc.qubit
    %q3 = dqc.alloc_qubit : !dqc.qubit

    // Stage 1: q0
    dqc.h %q0 : (!dqc.qubit)
    dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)
    dqc.rz %q1 1.5707963 : (!dqc.qubit)
    dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q0, %q2 : (!dqc.qubit, !dqc.qubit)
    dqc.rz %q2 0.7853981 : (!dqc.qubit)
    dqc.cnot %q0, %q2 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q0, %q3 : (!dqc.qubit, !dqc.qubit)
    dqc.rz %q3 0.3926990 : (!dqc.qubit)
    dqc.cnot %q0, %q3 : (!dqc.qubit, !dqc.qubit)

    // Stage 2: q1
    dqc.h %q1 : (!dqc.qubit)
    dqc.cnot %q1, %q2 : (!dqc.qubit, !dqc.qubit)
    dqc.rz %q2 1.5707963 : (!dqc.qubit)
    dqc.cnot %q1, %q2 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q1, %q3 : (!dqc.qubit, !dqc.qubit)
    dqc.rz %q3 0.7853981 : (!dqc.qubit)
    dqc.cnot %q1, %q3 : (!dqc.qubit, !dqc.qubit)

    // Stage 3: q2
    dqc.h %q2 : (!dqc.qubit)
    dqc.cnot %q2, %q3 : (!dqc.qubit, !dqc.qubit)
    dqc.rz %q3 1.5707963 : (!dqc.qubit)
    dqc.cnot %q2, %q3 : (!dqc.qubit, !dqc.qubit)

    // Stage 4: q3
    dqc.h %q3 : (!dqc.qubit)

    // Bit-reversal
    dqc.swap %q0, %q3 : (!dqc.qubit, !dqc.qubit)
    dqc.swap %q1, %q2 : (!dqc.qubit, !dqc.qubit)

    return
  }
}

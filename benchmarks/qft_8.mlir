// Quantum Fourier Transform (8 qubits)
// Simplified: H + nearest-neighbor CNOT-Rz decomposition of controlled rotations
module {
  func.func @qft_8() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit
    %q2 = dqc.alloc_qubit : !dqc.qubit
    %q3 = dqc.alloc_qubit : !dqc.qubit
    %q4 = dqc.alloc_qubit : !dqc.qubit
    %q5 = dqc.alloc_qubit : !dqc.qubit
    %q6 = dqc.alloc_qubit : !dqc.qubit
    %q7 = dqc.alloc_qubit : !dqc.qubit

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
    dqc.cnot %q2, %q4 : (!dqc.qubit, !dqc.qubit)
    dqc.rz %q4 0.7853981 : (!dqc.qubit)
    dqc.cnot %q2, %q4 : (!dqc.qubit, !dqc.qubit)

    // Stage 4: q3
    dqc.h %q3 : (!dqc.qubit)
    dqc.cnot %q3, %q4 : (!dqc.qubit, !dqc.qubit)
    dqc.rz %q4 1.5707963 : (!dqc.qubit)
    dqc.cnot %q3, %q4 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q3, %q5 : (!dqc.qubit, !dqc.qubit)
    dqc.rz %q5 0.7853981 : (!dqc.qubit)
    dqc.cnot %q3, %q5 : (!dqc.qubit, !dqc.qubit)

    // Stage 5: q4
    dqc.h %q4 : (!dqc.qubit)
    dqc.cnot %q4, %q5 : (!dqc.qubit, !dqc.qubit)
    dqc.rz %q5 1.5707963 : (!dqc.qubit)
    dqc.cnot %q4, %q5 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q4, %q6 : (!dqc.qubit, !dqc.qubit)
    dqc.rz %q6 0.7853981 : (!dqc.qubit)
    dqc.cnot %q4, %q6 : (!dqc.qubit, !dqc.qubit)

    // Stage 6: q5
    dqc.h %q5 : (!dqc.qubit)
    dqc.cnot %q5, %q6 : (!dqc.qubit, !dqc.qubit)
    dqc.rz %q6 1.5707963 : (!dqc.qubit)
    dqc.cnot %q5, %q6 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q5, %q7 : (!dqc.qubit, !dqc.qubit)
    dqc.rz %q7 0.7853981 : (!dqc.qubit)
    dqc.cnot %q5, %q7 : (!dqc.qubit, !dqc.qubit)

    // Stage 7: q6
    dqc.h %q6 : (!dqc.qubit)
    dqc.cnot %q6, %q7 : (!dqc.qubit, !dqc.qubit)
    dqc.rz %q7 1.5707963 : (!dqc.qubit)
    dqc.cnot %q6, %q7 : (!dqc.qubit, !dqc.qubit)

    // Stage 8: q7
    dqc.h %q7 : (!dqc.qubit)

    // Bit-reversal SWAPs
    dqc.swap %q0, %q7 : (!dqc.qubit, !dqc.qubit)
    dqc.swap %q1, %q6 : (!dqc.qubit, !dqc.qubit)
    dqc.swap %q2, %q5 : (!dqc.qubit, !dqc.qubit)
    dqc.swap %q3, %q4 : (!dqc.qubit, !dqc.qubit)

    return
  }
}

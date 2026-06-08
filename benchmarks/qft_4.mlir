// Quantum Fourier Transform (4 qubits)
// QFT applies H + controlled rotations in a cascade pattern
// QFT_4 = H(q0) Rz(pi/2,q1|q0) Rz(pi/4,q2|q0) Rz(pi/8,q3|q0)
//          H(q1) Rz(pi/2,q2|q1) Rz(pi/4,q3|q1)
//          H(q2) Rz(pi/2,q3|q2)
//          H(q3) + SWAP(q0,q3) SWAP(q1,q2)
// Note: controlled rotations approximated as CNOT + Rz decomposition
module {
  func.func @qft_4() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit
    %q2 = dqc.alloc_qubit : !dqc.qubit
    %q3 = dqc.alloc_qubit : !dqc.qubit

    // Stage 1: q0
    dqc.h %q0 : (!dqc.qubit)
    // CR(pi/2) on q1 controlled by q0 ≈ CNOT + Rz
    dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)
    dqc.rz %q1 1.5707963 : (!dqc.qubit)
    dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)
    // CR(pi/4) on q2 controlled by q0
    dqc.cnot %q0, %q2 : (!dqc.qubit, !dqc.qubit)
    dqc.rz %q2 0.7853981 : (!dqc.qubit)
    dqc.cnot %q0, %q2 : (!dqc.qubit, !dqc.qubit)
    // CR(pi/8) on q3 controlled by q0
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

    // Bit-reversal SWAPs
    dqc.swap %q0, %q3 : (!dqc.qubit, !dqc.qubit)
    dqc.swap %q1, %q2 : (!dqc.qubit, !dqc.qubit)

    return
  }
}

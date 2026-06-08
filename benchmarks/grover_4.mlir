// Grover's Search Algorithm (4 qubits, 1 iteration)
// Searches for marked state |1011>
// Structure: Initialization → Oracle → Diffusion
module {
  func.func @grover_4() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit
    %q2 = dqc.alloc_qubit : !dqc.qubit
    %q3 = dqc.alloc_qubit : !dqc.qubit

    // === Initialization: H on all qubits ===
    dqc.h %q0 : (!dqc.qubit)
    dqc.h %q1 : (!dqc.qubit)
    dqc.h %q2 : (!dqc.qubit)
    dqc.h %q3 : (!dqc.qubit)

    // === Oracle for |1011>: flip phase of target state ===
    // Apply X to q2 (the 0-bit in our target)
    dqc.x %q2 : (!dqc.qubit)
    // Multi-controlled Z via Toffoli decomposition
    dqc.h %q3 : (!dqc.qubit)
    dqc.ccx %q0, %q1, %q3 : (!dqc.qubit, !dqc.qubit, !dqc.qubit)
    dqc.ccx %q2, %q3, %q0 : (!dqc.qubit, !dqc.qubit, !dqc.qubit)
    dqc.h %q3 : (!dqc.qubit)
    // Undo X on q2
    dqc.x %q2 : (!dqc.qubit)

    // === Diffusion operator: 2|s><s| - I ===
    dqc.h %q0 : (!dqc.qubit)
    dqc.h %q1 : (!dqc.qubit)
    dqc.h %q2 : (!dqc.qubit)
    dqc.h %q3 : (!dqc.qubit)

    dqc.x %q0 : (!dqc.qubit)
    dqc.x %q1 : (!dqc.qubit)
    dqc.x %q2 : (!dqc.qubit)
    dqc.x %q3 : (!dqc.qubit)

    // Multi-controlled Z
    dqc.h %q3 : (!dqc.qubit)
    dqc.ccx %q0, %q1, %q3 : (!dqc.qubit, !dqc.qubit, !dqc.qubit)
    dqc.h %q3 : (!dqc.qubit)

    dqc.x %q0 : (!dqc.qubit)
    dqc.x %q1 : (!dqc.qubit)
    dqc.x %q2 : (!dqc.qubit)
    dqc.x %q3 : (!dqc.qubit)

    dqc.h %q0 : (!dqc.qubit)
    dqc.h %q1 : (!dqc.qubit)
    dqc.h %q2 : (!dqc.qubit)
    dqc.h %q3 : (!dqc.qubit)

    return
  }
}

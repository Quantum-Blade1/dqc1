// GHZ State Preparation (8 qubits)
// Creates the state (|00000000> + |11111111>) / sqrt(2)
module {
  func.func @ghz_8() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit
    %q2 = dqc.alloc_qubit : !dqc.qubit
    %q3 = dqc.alloc_qubit : !dqc.qubit
    %q4 = dqc.alloc_qubit : !dqc.qubit
    %q5 = dqc.alloc_qubit : !dqc.qubit
    %q6 = dqc.alloc_qubit : !dqc.qubit
    %q7 = dqc.alloc_qubit : !dqc.qubit

    dqc.h %q0 : (!dqc.qubit)
    dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q1, %q2 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q2, %q3 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q3, %q4 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q4, %q5 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q5, %q6 : (!dqc.qubit, !dqc.qubit)
    dqc.cnot %q6, %q7 : (!dqc.qubit, !dqc.qubit)

    return
  }
}

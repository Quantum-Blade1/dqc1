// Example 12: Reset + Multi-Controlled X
//
// Demonstrates qubit reset (measure + conditional X) and MCX gate.
// Prepares |111> on 3 qubits, uses MCX (Toffoli) to flip a target.

module {
  func.func @reset_and_mcx() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit
    %q2 = dqc.alloc_qubit : !dqc.qubit
    %q3 = dqc.alloc_qubit : !dqc.qubit

    // Prepare |1110> (q0=1, q1=1, q2=1, q3=0)
    dqc.x %q0 : (!dqc.qubit)
    dqc.x %q1 : (!dqc.qubit)
    dqc.x %q2 : (!dqc.qubit)

    // MCX: flip q3 if q0, q1, q2 are all |1> => q3 becomes |1>
    dqc.mcx %q0, %q1, %q2, %q3 : (!dqc.qubit, !dqc.qubit, !dqc.qubit, !dqc.qubit)

    // Now state is |1111>. Reset q0 back to |0>
    dqc.reset %q0 : (!dqc.qubit)

    // Final state: |0111> (q0=0, q1=1, q2=1, q3=1)
    return
  }
}

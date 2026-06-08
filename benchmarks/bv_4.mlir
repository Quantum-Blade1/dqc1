// Bernstein-Vazirani Algorithm (4 qubits)
// Finds hidden string s = 1011
// Circuit: H on all, oracle CNOTs for bits of s, H on all, measure
module {
  func.func @bv_4() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit
    %q2 = dqc.alloc_qubit : !dqc.qubit
    %q3 = dqc.alloc_qubit : !dqc.qubit
    %anc = dqc.alloc_qubit : !dqc.qubit

    // Prepare ancilla in |-> state
    dqc.x %anc : (!dqc.qubit)
    dqc.h %anc : (!dqc.qubit)

    // Hadamard on query register
    dqc.h %q0 : (!dqc.qubit)
    dqc.h %q1 : (!dqc.qubit)
    dqc.h %q2 : (!dqc.qubit)
    dqc.h %q3 : (!dqc.qubit)

    // Oracle for s = 1011: CNOT from bits where s_i = 1
    dqc.cnot %q0, %anc : (!dqc.qubit, !dqc.qubit)  // s_0 = 1
    dqc.cnot %q1, %anc : (!dqc.qubit, !dqc.qubit)  // s_1 = 1
    // s_2 = 0: no CNOT
    dqc.cnot %q3, %anc : (!dqc.qubit, !dqc.qubit)  // s_3 = 1

    // Hadamard on query register
    dqc.h %q0 : (!dqc.qubit)
    dqc.h %q1 : (!dqc.qubit)
    dqc.h %q2 : (!dqc.qubit)
    dqc.h %q3 : (!dqc.qubit)

    // Measure
    %c0 = dqc.measure %q0 : (!dqc.qubit) -> !dqc.cbit
    %c1 = dqc.measure %q1 : (!dqc.qubit) -> !dqc.cbit
    %c2 = dqc.measure %q2 : (!dqc.qubit) -> !dqc.cbit
    %c3 = dqc.measure %q3 : (!dqc.qubit) -> !dqc.cbit

    return
  }
}

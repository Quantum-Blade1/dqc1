// Quantum Teleportation (3 qubits)
// Teleports the state of q0 to q2 using entangled pair (q1, q2)
// q0: data qubit, q1+q2: Bell pair
module {
  func.func @teleport() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit
    %q2 = dqc.alloc_qubit : !dqc.qubit

    // Prepare data qubit in superposition (something to teleport)
    dqc.rx %q0 1.0471975 : (!dqc.qubit)

    // Create Bell pair between q1 and q2
    dqc.h %q1 : (!dqc.qubit)
    dqc.cnot %q1, %q2 : (!dqc.qubit, !dqc.qubit)

    // Bell measurement on q0,q1
    dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)
    dqc.h %q0 : (!dqc.qubit)
    %c0 = dqc.measure %q0 : (!dqc.qubit) -> !dqc.cbit
    %c1 = dqc.measure %q1 : (!dqc.qubit) -> !dqc.cbit

    // Corrections (classically controlled, here we apply unconditionally
    // as a demonstration — real teleportation needs classical feedback)
    dqc.x %q2 : (!dqc.qubit)
    dqc.z %q2 : (!dqc.qubit)

    return
  }
}

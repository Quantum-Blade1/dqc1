// Example 7: Quantum Teleportation
//
// The famous quantum teleportation protocol!
// Transfers the quantum state of q0 to q2 using entanglement
//
// How it works:
//   1. Prepare q0 in some state (Rx rotation)
//   2. Create Bell pair between q1 and q2
//   3. Bell measurement on q0,q1
//   4. Apply corrections to q2
//   Result: q2 now has the original state of q0!
//
// The DQC compiler will automatically distribute this across QPUs

module {
  func.func @teleportation() {
    %q0 = dqc.alloc_qubit : !dqc.qubit  // data qubit (state to teleport)
    %q1 = dqc.alloc_qubit : !dqc.qubit  // entangled pair (Alice's half)
    %q2 = dqc.alloc_qubit : !dqc.qubit  // entangled pair (Bob's half)

    // Prepare an interesting state to teleport
    dqc.rx %q0 1.0471975 : (!dqc.qubit)

    // Create Bell pair between q1 and q2
    dqc.h %q1 : (!dqc.qubit)
    dqc.cnot %q1, %q2 : (!dqc.qubit, !dqc.qubit)

    // Bell measurement
    dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)
    dqc.h %q0 : (!dqc.qubit)
    %c0 = dqc.measure %q0 : (!dqc.qubit) -> !dqc.cbit
    %c1 = dqc.measure %q1 : (!dqc.qubit) -> !dqc.cbit

    // Corrections (applied unconditionally for demo)
    dqc.x %q2 : (!dqc.qubit)
    dqc.z %q2 : (!dqc.qubit)

    return
  }
}

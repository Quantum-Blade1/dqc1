// Example 11: Mid-Circuit Measurement + Classical Feedback
//
// Demonstrates quantum teleportation with proper corrections:
// 1. Create entangled pair (q1, q2)
// 2. Prepare state to teleport on q0
// 3. Bell measurement on (q0, q1)
// 4. Conditionally apply X and Z corrections on q2
//
// This is impossible without classical feedback (c_if)

module {
  func.func @teleport_with_corrections() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit
    %q2 = dqc.alloc_qubit : !dqc.qubit

    // Prepare state to teleport: |+> = H|0>
    dqc.h %q0 : (!dqc.qubit)

    // Create Bell pair between q1 and q2
    dqc.h %q1 : (!dqc.qubit)
    dqc.cnot %q1, %q2 : (!dqc.qubit, !dqc.qubit)

    // Bell measurement on (q0, q1)
    dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)
    dqc.h %q0 : (!dqc.qubit)

    %c0 = dqc.measure %q0 : (!dqc.qubit) -> !dqc.cbit
    %c1 = dqc.measure %q1 : (!dqc.qubit) -> !dqc.cbit

    // Classical corrections on q2 based on measurement outcomes
    dqc.c_if %c1 {
      dqc.x %q2 : (!dqc.qubit)
    }
    dqc.c_if %c0 {
      dqc.z %q2 : (!dqc.qubit)
    }

    return
  }
}

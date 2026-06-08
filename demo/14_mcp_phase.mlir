// Example 14: Multi-Controlled Phase Gate
//
// Applies a controlled-controlled-Z (CCZ) using mcp with angle=pi.
// Prepares |11+> and applies CCZ, which adds phase -1 to |111>.
// Then applies H to last qubit to observe the phase kickback.

module {
  func.func @mcp_demo() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit
    %q2 = dqc.alloc_qubit : !dqc.qubit

    // Prepare |11+> = |11> tensor H|0>
    dqc.x %q0 : (!dqc.qubit)
    dqc.x %q1 : (!dqc.qubit)
    dqc.h %q2 : (!dqc.qubit)

    // CCZ = multi-controlled phase(pi) on q2, controlled by q0, q1
    // This flips the phase of |111> component: |11+> -> |11->
    dqc.mcp %q0, %q1, %q2 3.14159265358979 : (!dqc.qubit, !dqc.qubit, !dqc.qubit)

    // H on q2 converts |-> back to |1>
    dqc.h %q2 : (!dqc.qubit)

    // Final state should be |111>
    return
  }
}

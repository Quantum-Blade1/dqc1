// Example 13: Repeat Loop
//
// Applies H gate 4 times to a qubit. H^2 = I, so H^4 = I.
// Starting from |0>, the final state should be |0>.

module {
  func.func @repeat_demo() {
    %q0 = dqc.alloc_qubit : !dqc.qubit

    // Apply H four times: H^4 = I
    dqc.repeat 4 {
      dqc.h %q0 : (!dqc.qubit)
    }

    // q0 should be back to |0>
    return
  }
}

// Example 5: Rotation Gates — Precise Quantum Control
//
// Unlike H (which gives exactly 50/50), rotation gates let you
// dial in ANY probability you want.
//
// Ry(angle) rotates the qubit state:
//   Ry(0)      → |0> with 100% probability
//   Ry(pi/2)   → 50/50 (same as Hadamard)
//   Ry(pi)     → |1> with 100% probability
//
// Here we use Ry(pi/3) ≈ 1.047 to get 25% |0> and 75% |1>

module {
  func.func @rotation_demo() {
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit
    %q2 = dqc.alloc_qubit : !dqc.qubit

    // 25% |1> probability
    dqc.ry %q0 1.0471975 : (!dqc.qubit)

    // 50% |1> probability (like Hadamard)
    dqc.ry %q1 1.5707963 : (!dqc.qubit)

    // 75% |1> probability
    dqc.ry %q2 2.0943951 : (!dqc.qubit)

    return
  }
}

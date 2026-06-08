// Example 1: Single Qubit — The Basics
//
// Start with 1 qubit in |0>, apply Hadamard to get superposition
// Expected output: |0> and |1> each with 50% probability
//
// This is the "Hello World" of quantum computing!

module {
  func.func @single_qubit() {
    %q = dqc.alloc_qubit : !dqc.qubit

    // Hadamard puts qubit into superposition: |0> → (|0> + |1>) / sqrt(2)
    dqc.h %q : (!dqc.qubit)

    return
  }
}

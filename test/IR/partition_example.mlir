// Partitioning example
module {
  func.func @partition_demo() {
    %q0 = dqc.qubit_alloc 0 : !dqc.qubit
    %q1 = dqc.qubit_alloc 1 : !dqc.qubit
    // Example multi-qubit gate that spans QPUs
    // (synthetic - for partitioning analysis)
    return
  }
}

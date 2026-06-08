// Partitioning example
module {
  func.func @partition_demo() {
    // Use EPR alloc/consume to create a simple cross-QPU interaction
    %epr = dqc.epr_alloc 0, 1 : !dqc.epr_handle
    dqc.epr_consume %epr : !dqc.epr_handle
    return
  }
}

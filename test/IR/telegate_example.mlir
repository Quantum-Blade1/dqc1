// TeleGate synthesis example
module {
  func.func @telegate_demo() {
    // Allocate EPR between QPUs 0 and 1
    %epr = dqc.epr_alloc 0, 1 : !dqc.epr_handle
    // Placeholder remote operation
    return
  }
}

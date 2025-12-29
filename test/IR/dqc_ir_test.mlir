//===- dqc_ir_test.mlir - DQC Dialect IR Tests ---*- MLIR -*-===//
//
// Tests for DQC dialect operations and types
//
//===----------------------------------------------------------===//

// Test: EPR allocation
func.func @test_epr_alloc() {
  // Allocate an entangled pair between QPUs 0 and 1
  %epr = dqc.epr_alloc 0, 1 : !dqc.epr_handle
  return
}

// Test: TeleGate operation (placeholder - would require QUIR qubits)
func.func @test_telegate() {
  return
}

// Test: Partition metadata
func.func @test_partition_info() {
  return
}

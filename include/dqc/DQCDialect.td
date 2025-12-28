// DQC Dialect: Distributed Quantum Computing dialect
// Defines operations for entanglement allocation (epr_alloc) and teleportation-based gates (telegate)

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// DQC Dialect Definition
//===----------------------------------------------------------------------===//

def DQC_Dialect : Dialect {
  let name = "dqc";
  let description = [{
    The DQC (Distributed Quantum Computing) dialect provides operations for
    managing distributed quantum computations across multiple QPUs, including
    entanglement allocation and teleported gate sequences.
  }];
  let cppNamespace = "::dqc";
  let hasConstantMaterializer = 1;
  let usePropertiesForAttributes = 0;
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

def DQC_QubitType : Type<CPred<"llvm::isa<dqc::QubitType>($_self)">,
                          "qubit">;

def DQC_EPRHandleType : Type<CPred<"llvm::isa<dqc::EPRHandleType>($_self)">,
                               "epr_handle">;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

class DQC_Op<string mnemonic, list<Trait> traits = []> :
    Op<DQC_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// epr_alloc Operation
// Allocates a shared entangled pair (Bell state) between two QPUs
//===----------------------------------------------------------------------===//

def DQC_EPRAllocOp : DQC_Op<"epr_alloc", [Pure]> {
  let summary = "Allocate a shared entangled pair (Bell state) between two QPUs";
  let description = [{
    Allocates a maximally entangled pair (Bell state) across two remote QPUs.
    This is the fundamental resource for quantum teleportation-based gates.
    
    Example:
    ```mlir
    %epr = dqc.epr_alloc %qpu0, %qpu1 : i32, i32 -> !dqc.epr_handle
    ```
  }];
  
  let arguments = (ins
    I32Attr:$source_qpu,
    I32Attr:$target_qpu,
    OptionalAttr<F32Attr>:$fidelity  // Optional: fidelity estimate of the link
  );
  let results = (outs DQC_EPRHandleType:$epr_handle);
  let assemblyFormat = "$source_qpu `,` $target_qpu attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// telegate Operation
// Performs a non-local CNOT via quantum teleportation
//===----------------------------------------------------------------------===//

def DQC_TeleGateOp : DQC_Op<"telegate", [SameVariadicOperandSize]> {
  let summary = "Perform a distributed gate via quantum teleportation";
  let description = [{
    Replaces a non-local two-qubit gate (primarily CNOT) with a teleportation
    sequence. Requires a pre-allocated EPR pair between the control and target QPUs.
    
    Sequence:
    1. Bell measurement on (control_qubit, epr_qubit_a)
    2. Measurement outcomes sent as classical bits to target QPU
    3. Target QPU applies corrective gate to (epr_qubit_b, target_qubit)
    
    Example:
    ```mlir
    %result = dqc.telegate %control, %target, %epr_handle 
                          : !quir.qubit, !quir.qubit, !dqc.epr_handle
                          -> !quir.qubit
    ```
  }];
  
  let arguments = (ins
    DQC_QubitType:$control_qubit,
    DQC_QubitType:$target_qubit,
    DQC_EPRHandleType:$epr_handle,
    I32Attr:$control_qpu,
    I32Attr:$target_qpu
  );
  let results = (outs DQC_QubitType:$result);
  let assemblyFormat = "$control_qubit `,` $target_qubit `,` $epr_handle attr-dict"
                      " `:` type(operands) `->` type(results)";
}

//===----------------------------------------------------------------------===//
// telegate_multi Operation
// Batch teleportation gate for multiple gates sharing the same control qubit (gate packing)
//===----------------------------------------------------------------------===//

def DQC_TeleGateMultiOp : DQC_Op<"telegate_multi", [SameVariadicOperandSize]> {
  let summary = "Perform multiple distributed gates sharing a single EPR pair";
  let description = [{
    Optimized variant of telegate that applies multiple gates (gate packet)
    that share a common control qubit using a single EPR pair allocation.
    This reduces e-bit consumption by up to 30%.
  }];
  
  let arguments = (ins
    DQC_QubitType:$control_qubit,
    Variadic<DQC_QubitType>:$target_qubits,
    DQC_EPRHandleType:$epr_handle,
    I32Attr:$control_qpu,
    I32ArrayAttr:$target_qpus
  );
  let results = (outs Variadic<DQC_QubitType>:$results);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// partition_info Operation (Metadata)
// Stores qubit-to-QPU mapping information
//===----------------------------------------------------------------------===//

def DQC_PartitionInfoOp : DQC_Op<"partition_info", [Pure]> {
  let summary = "Metadata operation storing qubit-to-QPU partition mapping";
  let description = [{
    This operation stores the result of the hypergraph partitioning phase.
    It maps logical qubit IDs to physical QPU IDs.
    
    Example:
    ```mlir
    dqc.partition_info {
      qubit_0 = 0,
      qubit_1 = 0,
      qubit_2 = 1,
      qubit_3 = 1
    }
    ```
  }];
  
  let arguments = (ins
    DictionaryAttr:$partition_map
  );
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// epr_consume Operation
// Marks EPR pair consumption and synchronization point
//===----------------------------------------------------------------------===//

def DQC_EPRConsumeOp : DQC_Op<"epr_consume", []> {
  let summary = "Mark EPR pair consumption and enforce synchronization";
  let description = [{
    Explicit operation denoting that an EPR pair has been consumed.
    Useful for tracking resource usage and enforcing ordering constraints.
  }];
  
  let arguments = (ins
    DQC_EPRHandleType:$epr_handle
  );
  let assemblyFormat = "$epr_handle attr-dict `:` type(operands)";
}

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
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

def DQC_QubitType : Type<CPred<"llvm::isa<dqc::QubitType>($_self)">,
                          "qubit">;

def DQC_EPRHandleType : Type<CPred<"llvm::isa<dqc::EPRHandleType>($_self)">,
                               "epr_handle">;

def DQC_CbitType : Type<CPred<"llvm::isa<dqc::CbitType>($_self)">,
                         "cbit">;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

class DQC_Op<string mnemonic, list<Trait> traits = []> :
    Op<DQC_Dialect, mnemonic, traits>;

// Qubit allocation
def DQC_AllocQubitOp : DQC_Op<"alloc_qubit", [Pure]> {
  let summary = "Allocate a qubit";
  let results = (outs DQC_QubitType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

// Single-qubit gates
def DQC_HOp  : DQC_Op<"h">  { let arguments = (ins DQC_QubitType:$qubit); let assemblyFormat = "$qubit attr-dict `:` `(` type($qubit) `)`"; }
def DQC_XOp  : DQC_Op<"x">  { let arguments = (ins DQC_QubitType:$qubit); let assemblyFormat = "$qubit attr-dict `:` `(` type($qubit) `)`"; }
def DQC_YOp  : DQC_Op<"y">  { let arguments = (ins DQC_QubitType:$qubit); let assemblyFormat = "$qubit attr-dict `:` `(` type($qubit) `)`"; }
def DQC_ZOp  : DQC_Op<"z">  { let arguments = (ins DQC_QubitType:$qubit); let assemblyFormat = "$qubit attr-dict `:` `(` type($qubit) `)`"; }
def DQC_SOp  : DQC_Op<"s">  { let arguments = (ins DQC_QubitType:$qubit); let assemblyFormat = "$qubit attr-dict `:` `(` type($qubit) `)`"; }
def DQC_TOp  : DQC_Op<"t">  { let arguments = (ins DQC_QubitType:$qubit); let assemblyFormat = "$qubit attr-dict `:` `(` type($qubit) `)`"; }

// Parametric rotation gates
def DQC_RxOp : DQC_Op<"rx"> { let arguments = (ins DQC_QubitType:$qubit, F64Attr:$angle); let assemblyFormat = "$qubit $angle attr-dict `:` `(` type($qubit) `)`"; }
def DQC_RyOp : DQC_Op<"ry"> { let arguments = (ins DQC_QubitType:$qubit, F64Attr:$angle); let assemblyFormat = "$qubit $angle attr-dict `:` `(` type($qubit) `)`"; }
def DQC_RzOp : DQC_Op<"rz"> { let arguments = (ins DQC_QubitType:$qubit, F64Attr:$angle); let assemblyFormat = "$qubit $angle attr-dict `:` `(` type($qubit) `)`"; }

// Two-qubit gates
def DQC_CNOTOp : DQC_Op<"cnot"> { let arguments = (ins DQC_QubitType:$ctrl, DQC_QubitType:$target); let assemblyFormat = "$ctrl `,` $target attr-dict `:` `(` type($ctrl) `,` type($target) `)`"; }
def DQC_CZOp   : DQC_Op<"cz">   { let arguments = (ins DQC_QubitType:$ctrl, DQC_QubitType:$target); let assemblyFormat = "$ctrl `,` $target attr-dict `:` `(` type($ctrl) `,` type($target) `)`"; }
def DQC_SWAPOp  : DQC_Op<"swap"> { let arguments = (ins DQC_QubitType:$q0, DQC_QubitType:$q1); let assemblyFormat = "$q0 `,` $q1 attr-dict `:` `(` type($q0) `,` type($q1) `)`"; }

// Three-qubit gate
def DQC_CCXOp : DQC_Op<"ccx"> { let arguments = (ins DQC_QubitType:$c0, DQC_QubitType:$c1, DQC_QubitType:$target); let assemblyFormat = "$c0 `,` $c1 `,` $target attr-dict `:` `(` type($c0) `,` type($c1) `,` type($target) `)`"; }

// Measurement
def DQC_MeasureOp : DQC_Op<"measure"> {
  let arguments = (ins DQC_QubitType:$qubit);
  let results = (outs DQC_CbitType:$result);
  let assemblyFormat = "$qubit attr-dict `:` `(` type($qubit) `)` `->` type($result)";
}

// Local gate (placeholder for optimized local operations)
def DQC_LocalGateOp : DQC_Op<"local_gate"> {
  let arguments = (ins DQC_QubitType:$qubit);
  let assemblyFormat = "$qubit attr-dict `:` `(` type($qubit) `)`";
}

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

def DQC_TeleGateOp : DQC_Op<"telegate", []> {
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
  let results = (outs);
  let assemblyFormat = "$control_qubit `,` $target_qubit `,` $epr_handle attr-dict"
                      " `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// telegate_multi Operation
// Batch teleportation gate for multiple gates sharing the same control qubit (gate packing)
//===----------------------------------------------------------------------===//

def DQC_TeleGateMultiOp : DQC_Op<"telegate_multi", []> {
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

//===----------------------------------------------------------------------===//
// barrier Operation
// Enforces ordering constraints across the barrier
//===----------------------------------------------------------------------===//

def DQC_BarrierOp : DQC_Op<"barrier"> {
  let summary = "Reordering barrier / fence";
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Reset Operation
// Resets a qubit to |0> via measure + conditional X
//===----------------------------------------------------------------------===//

def DQC_ResetOp : DQC_Op<"reset"> {
  let summary = "Reset qubit to |0> state";
  let arguments = (ins DQC_QubitType:$qubit);
  let assemblyFormat = "$qubit attr-dict `:` `(` type($qubit) `)`";
}

//===----------------------------------------------------------------------===//
// Multi-Controlled X (generalized Toffoli)
//===----------------------------------------------------------------------===//

def DQC_MCXOp : DQC_Op<"mcx"> {
  let summary = "Multi-controlled X gate (generalized Toffoli)";
  let description = [{
    Applies X to the last qubit if all preceding qubits are |1>.
    Syntax: dqc.mcx %c0, %c1, ..., %target : (!dqc.qubit, ...)
  }];
  let arguments = (ins Variadic<DQC_QubitType>:$qubits);
  let hasCustomAssemblyFormat = 1;
}


//===- MPIDialect.td - MPI Dialect for Distributed Quantum --*- tablegen -*-===//
//
// Defines the MPI dialect ops used as a lowering target from the DQC
// teleportation primitives. These ops model distributed communication
// (EPR distribution, telegate sequencing) and are subsequently lowered
// to LLVM runtime calls.
//
//===----------------------------------------------------------------------===//

#ifndef MPI_DIALECT_TD
#define MPI_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def MPI_Dialect : Dialect {
  let name = "mpi";
  let cppNamespace = "::dqc::mpi";
  let summary = "Dialect for distributed quantum communication primitives";
}

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

def MPI_QubitType : Type<CPred<"::dqc::isQubitType($_self)">,
                         "qubit type">;

def MPI_EprHandleType : Type<CPred<"::dqc::isEprHandleType($_self)">,
                              "EPR handle type">;

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

class MPI_Op<string mnemonic, list<Trait> traits = []> :
    Op<MPI_Dialect, mnemonic, traits>;

def MPI_DistributeEprOp : MPI_Op<"distribute_epr"> {
  let summary = "Distribute an EPR pair between two QPUs";
  let arguments = (ins I32Attr:$src_qpu, I32Attr:$tgt_qpu);
  let results = (outs MPI_EprHandleType:$epr_handle);
  let assemblyFormat = "$src_qpu `,` $tgt_qpu attr-dict `:` type($epr_handle)";
}

def MPI_TeleGateSequenceOp : MPI_Op<"telegate_sequence"> {
  let summary = "Execute a telegate sequence between two qubits via EPR";
  let arguments = (ins
    MPI_QubitType:$ctrl,
    MPI_QubitType:$target,
    MPI_EprHandleType:$epr_handle,
    I32Attr:$control_qpu,
    I32Attr:$target_qpu
  );
  let assemblyFormat = [{
    $ctrl `,` $target `,` $epr_handle
    attr-dict `:` `(` type($ctrl) `,` type($target) `,` type($epr_handle) `)`
  }];
}

#endif // MPI_DIALECT_TD

//===- MPILoweringPass.cpp - DQC to MPI Lowering ---*- C++ -*-===//
//
// This file implements the MPI Lowering Pass, which converts the DQC dialect
// to the MPI dialect for distributed execution simulation.
//
// Phase D: MPI Lowering for Execution
//===--------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"


#define DEBUG_TYPE "mpi-lowering"

namespace {

/// Maps DQC operations to MPI communication patterns
class DQCToMPILowering {
public:
  static mlir::Operation *lowerEPRAlloc(mlir::Operation *op,
                                        mlir::PatternRewriter &rewriter) {
    // dqc.epr_alloc %src_qpu, %tgt_qpu
    // =>
    // We generate code that works in SPMD fashion.
    // Logic:
    // %rank = mpi.comm_rank()
    // if (%rank == %src_qpu) {
    //   %epr_half = dqc.create_epr_pair_half()
    //   mpi.send(%epr_half, %tgt_qpu, tag=0)
    // }
    // if (%rank == %tgt_qpu) {
    //   %epr_half = mpi.recv(%src_qpu, tag=0)
    // }

    // For MVP, we insert the raw MPI ops. The Rank Dispatcher pass will clean
    // up. However, to permit SSA value replacement, we need to produce a result
    // that represents the EPR handle. But the handle exists on two nodes? In
    // DQC IR, the handle is a single value used by TeleGate.

    // Simplified MVP implementation:
    // Replace dqc.epr_alloc with mpi.pair_distribution
    // This is a higher-level MPI op that encapsulates the Isend/Irecv logic

    mlir::Location loc = op->getLoc();

    // Create mpi.distribute_epr
    mlir::OperationState adaptState(loc, "mpi.distribute_epr");
    adaptState.addTypes(op->getResultTypes()); // Return EPR handle type
    adaptState.addAttributes(op->getAttrs());

    auto *mpi_op = rewriter.createOperation(adaptState);

    LLVM_DEBUG(llvm::dbgs() << "Lowering epr_alloc to mpi.distribute_epr\n");

    rewriter.replaceOp(op, mpi_op->getResults());
    return mpi_op;
  }

  static mlir::Operation *lowerTeleGate(mlir::Operation *op,
                                        mlir::PatternRewriter &rewriter) {
    // dqc.telegate %ctrl, %tgt, %epr
    // =>
    // Distributed CNOT via teleportation

    mlir::Location loc = op->getLoc();
    auto ctrl = op->getOperand(0);
    auto tgt = op->getOperand(1);
    auto epr = op->getOperand(2);

    // Create mpi.telegate_sequence
    // This op expands to:
    // 1. Local CNOT(ctrl, epr) @ source
    // 2. Measure(epr) -> bit @ source
    // 3. mpi.send(bit) source -> target
    // 4. mpi.recv(bit) @ target
    // 5. Apply correction @ target

    mlir::OperationState state(loc, "mpi.telegate_sequence");
    state.addTypes(op->getResultTypes());
    state.addOperands({ctrl, tgt, epr});
    state.addAttributes(op->getAttrs());

    auto *mpi_op = rewriter.createOperation(state);

    LLVM_DEBUG(llvm::dbgs() << "Lowering telegate to mpi.telegate_sequence\n");

    rewriter.replaceOp(op, mpi_op->getResults());
    return mpi_op;
  }
};

/// MPI Lowering Pass
class MPILoweringPass
    : public mlir::PassWrapper<MPILoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_OPNAME_ALLOCATIONFN(MPILoweringPass)

  StringRef getArgument() const final { return "dqc-mpi-lowering"; }
  StringRef getDescription() const final {
    return "Lower DQC dialect to MPI dialect for distributed execution";
  }

  MPILoweringPass() = default;
  MPILoweringPass(const MPILoweringPass &) {}

  Option<bool> generateSPMDKernel{
      *this, "generate-spmd",
      llvm::cl::desc("Generate SPMD kernel with MPI rank dispatch"),
      llvm::cl::init(true)};

  Option<int> numMPIRanks{
      *this, "num-ranks",
      llvm::cl::desc("Number of MPI ranks (= number of QPUs)"),
      llvm::cl::init(2)};

private:
  /// Generate dispatcher to split code by MPI rank
  void generateRankDispatcher(mlir::ModuleOp module, int num_ranks) {
    mlir::MLIRContext *ctx = module.getContext();
    mlir::Location loc = module.getLoc();

    // Create wrapper function that calls different code based on MPI rank
    // Pseudocode:
    // func @mpi_main() {
    //   %rank = mpi.comm_rank()
    //   switch (%rank)
    //     case 0: call @rank_0_kernel()
    //     case 1: call @rank_1_kernel()
    //     ...
    // }

    LLVM_DEBUG(llvm::dbgs()
               << "Generating SPMD dispatcher for " << num_ranks << " ranks\n");
  }

  /// Lower dqc.epr_alloc to MPI communication
  void lowerEPRAlloc(mlir::func::FuncOp func) {
    func.walk([](mlir::Operation *op) {
      // Look for dqc.epr_alloc operations
      // Replace with MPI isend/irecv pairs
      LLVM_DEBUG(llvm::dbgs() << "Processing EPR allocation in function\n");
    });
  }

  /// Lower dqc.telegate to MPI measurement + send/recv
  void lowerTeleGate(mlir::func::FuncOp func) {
    func.walk([](mlir::Operation *op) {
      // Look for dqc.telegate operations
      // Replace with measurement + mpi.send/recv sequence
      LLVM_DEBUG(llvm::dbgs() << "Processing teleportation gate in function\n");
    });
  }

  /// Partition code by MPI rank using dataflow analysis
  void partitionByRank(mlir::func::FuncOp func) {
    // Analyze which operations belong to which QPU/rank
    // Create separate code paths for each rank

    func.walk([](mlir::Operation *op) {
      // Extract rank information from operation attributes
      // Route to appropriate code section
    });
  }

public:
  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "Starting MPI lowering on module\n");
    LLVM_DEBUG(llvm::dbgs() << "Number of MPI ranks: " << numMPIRanks << "\n");

    // Generate SPMD dispatcher if requested
    if (generateSPMDKernel) {
      generateRankDispatcher(module, numMPIRanks);
    }

    // Process each function
    module.walk([&](mlir::func::FuncOp func) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Lowering function: " << func.getName() << "\n");

      // Lower EPR allocations
      lowerEPRAlloc(func);

      // Lower TeleGate operations
      lowerTeleGate(func);

      // Partition code by MPI rank
      partitionByRank(func);
    });

    LLVM_DEBUG(llvm::dbgs() << "MPI lowering completed\n");
  }
};

} // anonymous namespace

namespace mlir {
namespace dqc {

std::unique_ptr<mlir::Pass> createMPILoweringPass() {
  return std::make_unique<::MPILoweringPass>();
}

} // namespace dqc
} // namespace mlir

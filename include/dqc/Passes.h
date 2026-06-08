//===- DQCPasses.h - DQC Compiler Passes ---*- C++ -*-===//
//
// This file defines entry points for all DQC passes.
//
//===--------------------------------------------------===//

#ifndef DQC_PASSES_H
#define DQC_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>


namespace dqc {

/// Phase A: Create partition pass for hypergraph-based qubit distribution
std::unique_ptr<mlir::Pass> createInteractionGraphPass();

/// Phase B: Create telegate synthesis pass for converting inter-QPU gates
std::unique_ptr<mlir::Pass> createTeleGateSynthesisPass();

/// Phase B.5: Create CCX decomposition pass
std::unique_ptr<mlir::Pass> createCCXDecompositionPass();

/// Phase C: Create optimization pass for gate reordering
std::unique_ptr<mlir::Pass> createGreedyReorderingPass(bool verify = false);

/// Phase D: Create MPI lowering pass for distributed execution
std::unique_ptr<mlir::Pass> createMPILoweringPass();

/// Phase E: Create LLVM lowering pass for runtime code generation
std::unique_ptr<mlir::Pass> createLLVMLoweringPass();

/// Register all DQC passes
void registerDQCPasses();

} // namespace dqc

#endif // DQC_PASSES_H

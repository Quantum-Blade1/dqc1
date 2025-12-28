//===- DQCPasses.h - DQC Compiler Passes ---*- C++ -*-===//
//
// This file defines entry points for all DQC passes.
//
//===--------------------------------------------------===//

#ifndef DQC_PASSES_H
#define DQC_PASSES_H

#include <memory>

namespace mlir {
class Pass;
class OpPassManager;

namespace dqc {

/// Phase A: Create partition pass for hypergraph-based qubit distribution
std::unique_ptr<Pass> createInteractionGraphPass();

/// Phase B: Create telegate synthesis pass for converting inter-QPU gates
std::unique_ptr<Pass> createTeleGateSynthesisPass();

/// Phase C: Create optimization pass for gate reordering
std::unique_ptr<Pass> createGreedyReorderingPass();

/// Phase D: Create MPI lowering pass for distributed execution
std::unique_ptr<Pass> createMPILoweringPass();

/// Register all DQC passes
void registerDQCPasses();

}  // namespace dqc
}  // namespace mlir

#endif  // DQC_PASSES_H

//===- PassRegistry.cpp - DQC Pass Registration ---*- C++ -*-===//
//
// This file registers all DQC compiler passes.
//
//===----------------------------------------------------------===//

#include "mlir/Pass/PassRegistry.h"
#include "dqc/Passes.h"
#include "mlir/Pass/Pass.h"



namespace dqc {

void registerDQCPasses() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createInteractionGraphPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTeleGateSynthesisPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createCCXDecompositionPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createGreedyReorderingPass();
  });
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createMPILoweringPass(); });
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createLLVMLoweringPass(); });
}

} // namespace dqc

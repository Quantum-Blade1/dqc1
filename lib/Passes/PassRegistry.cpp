//===- PassRegistry.cpp - DQC Pass Registration ---*- C++ -*-===//
//
// This file registers all DQC compiler passes.
//
//===----------------------------------------------------------===//

#include "mlir/Pass/PassRegistry.h"
#include "dqc/Passes.h"
#include "mlir/Pass/Pass.h"

namespace dqc {
extern std::unique_ptr<mlir::Pass> createInteractionGraphPass();
extern std::unique_ptr<mlir::Pass> createTeleGateSynthesisPass();
extern std::unique_ptr<mlir::Pass> createGreedyReorderingPass();
extern std::unique_ptr<mlir::Pass> createMPILoweringPass();
} // namespace dqc

namespace dqc {

void registerDQCPasses() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createInteractionGraphPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTeleGateSynthesisPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createGreedyReorderingPass();
  });
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createMPILoweringPass(); });
}

} // namespace dqc

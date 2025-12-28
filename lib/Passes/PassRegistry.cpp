//===- PassRegistry.cpp - DQC Pass Registration ---*- C++ -*-===//
//
// This file registers all DQC compiler passes.
//
//===----------------------------------------------------------===//

#include "dqc/Passes.h"
#include "mlir/Pass/PassRegistry.h"

extern std::unique_ptr<mlir::Pass> createInteractionGraphPass();
extern std::unique_ptr<mlir::Pass> createTeleGateSynthesisPass();
extern std::unique_ptr<mlir::Pass> createGreedyReorderingPass();
extern std::unique_ptr<mlir::Pass> createMPILoweringPass();

namespace mlir {
namespace dqc {

void registerDQCPasses() {
  registerPass([]() -> std::unique_ptr<Pass> {
    return createInteractionGraphPass();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return createTeleGateSynthesisPass();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return createGreedyReorderingPass();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return createMPILoweringPass();
  });
}

}  // namespace dqc
}  // namespace mlir

//===- CCXDecompositionPass.cpp - CCX Decomposition Pass -------*- C++ -*-===//
//
// This file implements the CCX Decomposition Pass, which replaces every
// dqc.ccx operation with a 15-gate sequence using CNOT, H, and Rz rotations.
//
//===----------------------------------------------------------------------===//

#include "dqc/DQCDialect.h"
#include "dqc/DQCOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "llvm/Support/Debug.h"
#include <cmath>

#define DEBUG_TYPE "ccx-decomposition"

namespace {

class CCXDecompositionPass
    : public mlir::PassWrapper<CCXDecompositionPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  llvm::StringRef getArgument() const final { return "dqc-ccx-decomposition"; }
  llvm::StringRef getDescription() const final {
    return "Decompose CCX gates into elementary single and two-qubit gates";
  }

  void runOnOperation() final {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<mlir::Operation *, 16> ccx_ops;
    func.walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef().contains("ccx"))
        ccx_ops.push_back(op);
    });

    for (auto *op : ccx_ops) {
      if (op->getNumOperands() < 3) continue;

      mlir::Value c0 = op->getOperand(0);
      mlir::Value c1 = op->getOperand(1);
      mlir::Value target = op->getOperand(2);

      mlir::OpBuilder builder(op);
      mlir::Location loc = op->getLoc();

      double pi = 3.14159265358979323846;
      double t_angle = pi / 4.0;
      double tdg_angle = -pi / 4.0;

      // 1. H target
      builder.create<dqc::HOp>(loc, target);
      // 2. CNOT c1, target
      builder.create<dqc::CNOTOp>(loc, c1, target);
      // 3. Rz target, -pi/4
      builder.create<dqc::RzOp>(loc, target, builder.getF64FloatAttr(tdg_angle));
      // 4. CNOT c0, target
      builder.create<dqc::CNOTOp>(loc, c0, target);
      // 5. Rz target, pi/4
      builder.create<dqc::RzOp>(loc, target, builder.getF64FloatAttr(t_angle));
      // 6. CNOT c1, target
      builder.create<dqc::CNOTOp>(loc, c1, target);
      // 7. Rz target, -pi/4
      builder.create<dqc::RzOp>(loc, target, builder.getF64FloatAttr(tdg_angle));
      // 8. CNOT c0, target
      builder.create<dqc::CNOTOp>(loc, c0, target);
      // 9. Rz c1, pi/4
      builder.create<dqc::RzOp>(loc, c1, builder.getF64FloatAttr(t_angle));
      // 10. Rz target, pi/4
      builder.create<dqc::RzOp>(loc, target, builder.getF64FloatAttr(t_angle));
      // 11. H target
      builder.create<dqc::HOp>(loc, target);
      // 12. CNOT c0, c1
      builder.create<dqc::CNOTOp>(loc, c0, c1);
      // 13. Rz c0, pi/4
      builder.create<dqc::RzOp>(loc, c0, builder.getF64FloatAttr(t_angle));
      // 14. Rz c1, -pi/4
      builder.create<dqc::RzOp>(loc, c1, builder.getF64FloatAttr(tdg_angle));
      // 15. CNOT c0, c1
      builder.create<dqc::CNOTOp>(loc, c0, c1);

      op->erase();
    }
  }
};

} // anonymous namespace

namespace dqc {
std::unique_ptr<mlir::Pass> createCCXDecompositionPass() {
  return std::make_unique<::CCXDecompositionPass>();
}
} // namespace dqc

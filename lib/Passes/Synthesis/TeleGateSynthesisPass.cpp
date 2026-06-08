//===- TeleGateSynthesisPass.cpp - TeleGate Synthesis Pass ---*- C++ -*-===//
//
// This file implements the TeleGate Synthesis Pass, which replaces inter-QPU
// CNOT operations with distributed dqc.telegate sequences involving
// entanglement allocation and teleportation.
//
// Phase B: TeleGate Synthesis (Dialect Conversion Pass)
//===----------------------------------------------------------------------===//

#include "dqc/DQCDialect.h"
#include "dqc/DQCOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "telegate-synthesis"

namespace {

/// Simple mapping table that reads `dqc.partition` attribute produced by the
/// partitioning pass.
class MappingTable {
private:
  llvm::DenseMap<int, int> qubit_to_qpu;

public:
  void loadFromFunctionAttr(mlir::func::FuncOp func) {
    auto partition_attr =
        func->getAttrOfType<mlir::DictionaryAttr>("dqc.partition");
    if (!partition_attr) {
      LLVM_DEBUG(llvm::dbgs() << "No partition metadata found in function\n");
      return;
    }

    for (const auto &entry : partition_attr) {
      auto key_str = entry.getName().str();
      if (key_str.rfind("qubit_", 0) == 0) {
        int qubit_id = std::stoi(key_str.substr(6));
        if (auto value_attr = llvm::dyn_cast<mlir::IntegerAttr>(entry.getValue())) {
          int qpu_id = value_attr.getInt();
          qubit_to_qpu[qubit_id] = qpu_id;
        }
      }
    }
  }

  bool getQPUAssignment(int qubit_id, int &qpu_id) const {
    auto it = qubit_to_qpu.find(qubit_id);
    if (it != qubit_to_qpu.end()) {
      qpu_id = it->second;
      return true;
    }
    return false;
  }

  bool isLocalGate(int ctrl_qubit, int tgt_qubit) const {
    int ctrl_qpu, tgt_qpu;
    if (getQPUAssignment(ctrl_qubit, ctrl_qpu) &&
        getQPUAssignment(tgt_qubit, tgt_qpu)) {
      return ctrl_qpu == tgt_qpu;
    }
    return true; // Default to local if mapping not found
  }
};

// Helper: attempt to extract an integer qubit id from a Value or from op
static int extractQubitId(mlir::Value qubit, const llvm::DenseMap<mlir::Value, int> &qubit_map) {
  if (!qubit)
    return -1;
  if (auto *def = qubit.getDefiningOp()) {
    if (auto idAttr = def->getAttrOfType<mlir::IntegerAttr>("quir.qubit_id"))
      return idAttr.getInt();
    auto it = qubit_map.find(qubit);
    if (it != qubit_map.end())
      return it->second;
  }
  // If this is a block argument, try to use argument number as fallback
  if (auto ba = llvm::dyn_cast<mlir::BlockArgument>(qubit))
    return static_cast<int>(ba.getArgNumber());
  return -1;
}

/// TeleGate Synthesis Pass (simple implementation)
class TeleGateSynthesisPass
    : public mlir::PassWrapper<TeleGateSynthesisPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  llvm::StringRef getArgument() const final { return "dqc-telegate-synthesis"; }
  llvm::StringRef getDescription() const final {
    return "Replace inter-QPU gates with TeleGate sequences";
  }

  void runOnOperation() final {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext *ctx = func.getContext();

    // Map qubit values to sequential integer IDs
    llvm::DenseMap<mlir::Value, int> qubit_map;
    int count = 0;
    func.walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef().contains("alloc_qubit")) {
        if (op->getNumResults() > 0) {
          qubit_map[op->getResult(0)] = count++;
        }
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "Starting TeleGate synthesis on function "
                            << func.getName() << "\n");

    MappingTable mapping_table;
    mapping_table.loadFromFunctionAttr(func);

    // Decompose cross-QPU SWAP operations into CNOTs
    llvm::SmallVector<mlir::Operation *, 16> swap_ops;
    func.walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef().contains("swap"))
        swap_ops.push_back(op);
    });

    for (auto *op : swap_ops) {
      if (op->getNumOperands() < 2) continue;
      mlir::Value q0 = op->getOperand(0);
      mlir::Value q1 = op->getOperand(1);
      int q0_id = extractQubitId(q0, qubit_map);
      int q1_id = extractQubitId(q1, qubit_map);
      if (q0_id == -1 || q1_id == -1) continue;
      if (mapping_table.isLocalGate(q0_id, q1_id)) continue; // Keep local swaps local

      mlir::OpBuilder builder(op);
      mlir::Location loc = op->getLoc();
      builder.create<dqc::CNOTOp>(loc, q0, q1);
      builder.create<dqc::CNOTOp>(loc, q1, q0);
      builder.create<dqc::CNOTOp>(loc, q0, q1);
      op->erase();
    }

    // Collect candidate CNOT-like ops first to avoid iterator invalidation
    llvm::SmallVector<mlir::Operation *, 16> candidates;
    func.walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef().contains("cnot"))
        candidates.push_back(op);
    });

    for (auto *op : candidates) {
      if (op->getNumOperands() < 2)
        continue;

      mlir::Value ctrl = op->getOperand(0);
      mlir::Value tgt = op->getOperand(1);

      int ctrl_id = extractQubitId(ctrl, qubit_map);
      int tgt_id = extractQubitId(tgt, qubit_map);

      if (ctrl_id == -1) {
        if (auto cAttr = op->getAttrOfType<mlir::IntegerAttr>("control_id"))
          ctrl_id = cAttr.getInt();
      }
      if (tgt_id == -1) {
        if (auto tAttr = op->getAttrOfType<mlir::IntegerAttr>("target_id"))
          tgt_id = tAttr.getInt();
      }

      if (ctrl_id == -1 || tgt_id == -1)
        continue; // cannot reason about mapping

      if (mapping_table.isLocalGate(ctrl_id, tgt_id))
        continue; // leave local gates alone

      int ctrl_qpu, tgt_qpu;
      mapping_table.getQPUAssignment(ctrl_id, ctrl_qpu);
      mapping_table.getQPUAssignment(tgt_id, tgt_qpu);

      mlir::OpBuilder builder(op);
      mlir::Location loc = op->getLoc();

      // Create EPR allocation
      mlir::OperationState eprState(loc, "dqc.epr_alloc");
      eprState.addTypes(dqc::EPRHandleType::get(ctx));
      eprState.addAttribute("source_qpu", builder.getI32IntegerAttr(ctrl_qpu));
      eprState.addAttribute("target_qpu", builder.getI32IntegerAttr(tgt_qpu));
      auto *eprAlloc = mlir::Operation::create(eprState);
      builder.getBlock()->getOperations().insert(builder.getInsertionPoint(),
                    eprAlloc);

      // Create telegate operation
      mlir::OperationState telegateState(loc, "dqc.telegate");
      for (auto t : op->getResultTypes())
        telegateState.addTypes(t);
      telegateState.addOperands(mlir::ValueRange{ctrl, tgt,
                      eprAlloc->getResult(0)});
      telegateState.addAttribute("control_qpu", builder.getI32IntegerAttr(ctrl_qpu));
      telegateState.addAttribute("target_qpu", builder.getI32IntegerAttr(tgt_qpu));
      auto *telegateOp = mlir::Operation::create(telegateState);
      builder.getBlock()->getOperations().insert(builder.getInsertionPoint(),
                    telegateOp);

      // Replace uses of original op results with telegate results
      auto n = std::min(op->getNumResults(), telegateOp->getNumResults());
      for (unsigned i = 0; i < n; ++i)
        op->getResult(i).replaceAllUsesWith(telegateOp->getResult(i));

      op->erase();
    }

    LLVM_DEBUG(llvm::dbgs() << "TeleGate synthesis completed\n");
  }
};

} // anonymous namespace

namespace dqc {

std::unique_ptr<mlir::Pass> createTeleGateSynthesisPass() {
  return std::make_unique<::TeleGateSynthesisPass>();
}

} // namespace dqc

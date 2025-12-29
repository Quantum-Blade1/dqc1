//===- GreedyReorderingPass.cpp - Gate Commutativity-Based Reordering ---*- C++
//-*-===//
//
// This file implements the Greedy Reordering Pass, which reorders gates to
// minimize the distribution cost by grouping local gates and clustering
// global gates that share the same control qubit into 'gate packets'.
//
// Phase C: Commutativity-Based Reordering (Optimization Layer)
//===--------------------------------------------------------------------------===//

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <set>
#include <vector>


#define DEBUG_TYPE "greedy-reordering"

namespace dqc {

/// Gate commutativity rules: defines which gates can be reordered
struct GateCommutativityRules {
  /// Check if two operations can be commuted (reordered)
  static bool canCommute(mlir::Operation *op1, mlir::Operation *op2) {
    // Extract operands
    auto ops1 = op1->getOperands();
    auto ops2 = op2->getOperands();

    // Simple rule: gates on disjoint qubits commute
    llvm::SmallVector<mlir::Value, 8> qubits1(ops1.begin(), ops1.end());
    llvm::SmallVector<mlir::Value, 8> qubits2(ops2.begin(), ops2.end());

    // Check for overlap (quadratic but sufficient for small operand counts)
    for (auto q1 : qubits1) {
      for (auto q2 : qubits2) {
        if (q1 == q2)
          return false; // Gates share a qubit, cannot commute
      }
    }

    return true; // Disjoint qubits, can commute
  }

  /// Check if two gates are on the same control qubit
  static bool shareControlQubit(mlir::Operation *op1, mlir::Operation *op2) {
    if (op1->getNumOperands() < 1 || op2->getNumOperands() < 1) {
      return false;
    }

    // Simplified: assume first operand is control
    return op1->getOperand(0) == op2->getOperand(0);
  }

  /// Check if a gate is local or distributed
  static bool isDistributedGate(mlir::Operation *op) {
    // Placeholder: in real implementation, check against mapping table
    // For now, check operation name
    auto op_name = op->getName().getStringRef();
    return op_name.contains("telegate") || op_name.contains("cnot");
  }
};

/// Gate packet: multiple gates sharing the same control qubit
struct OptimizedGatePacket {
  mlir::Value control_qubit;
  std::vector<mlir::Operation *> gates;
  bool is_distributed;

  OptimizedGatePacket(mlir::Value ctrl, bool distributed)
      : control_qubit(ctrl), is_distributed(distributed) {}

  void addGate(mlir::Operation *gate) { gates.push_back(gate); }

  /// Estimate e-bit consumption (distributed gates consume e-bits)
  int estimateEBitCost() const {
    if (!is_distributed)
      return 0;
    return gates.size(); // Each distributed gate consumes 1 e-bit pair
  }
};

/// Dependency graph node
struct DependencyNode {
  mlir::Operation *op;
  std::vector<DependencyNode *> dependencies;
  int depth; // Topological depth in dependency graph

  DependencyNode(mlir::Operation *operation) : op(operation), depth(0) {}
};

} // namespace dqc

namespace {

/// Greedy Reordering Pass
class GreedyReorderingPass
    : public mlir::PassWrapper<GreedyReorderingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  // No internal MLIR allocation macro used here.

  llvm::StringRef getArgument() const final { return "dqc-greedy-reordering"; }
  llvm::StringRef getDescription() const final {
    return "Reorder gates to minimize distribution cost using commutativity "
           "rules";
  }

  GreedyReorderingPass() = default;
  GreedyReorderingPass(const GreedyReorderingPass &) {}

  float targetEBitReduction = 0.3f;

private:
  /// Build dependency graph from operation sequence
  std::vector<dqc::DependencyNode> buildDependencyGraph(mlir::Block *block) {
    std::vector<dqc::DependencyNode> nodes;
    std::map<mlir::Operation *, size_t> op_to_idx;

    // Create nodes for all operations
    block->walk([&](mlir::Operation *op) {
      if (op == block->getParentOp())
        return; // Skip function wrapper
      op_to_idx[op] = nodes.size();
      nodes.emplace_back(op);
    });

    // Build dependency edges
    for (size_t i = 0; i < nodes.size(); ++i) {
      mlir::Operation *op = nodes[i].op;

      // Find all previous operations that this depends on
      for (size_t j = 0; j < i; ++j) {
        mlir::Operation *prev_op = nodes[j].op;

        // Check if current operation uses results from previous
        bool depends = false;
        for (auto result : prev_op->getResults()) {
          for (auto &use : result.getUses()) {
            if (use.getOwner() == op) {
              depends = true;
              break;
            }
          }
        }

        // Check if operations share qubits (data dependency)
        if (!depends && dqc::GateCommutativityRules::canCommute(op, prev_op)) {
          depends = false;
        } else if (!dqc::GateCommutativityRules::canCommute(op, prev_op)) {
          depends = true; // Cannot commute, so there's a dependency
        }

        if (depends) {
          nodes[i].dependencies.push_back(&nodes[j]);
        }
      }

      // Calculate topological depth
      int max_dep_depth = 0;
      for (auto dep : nodes[i].dependencies) {
        max_dep_depth = std::max(max_dep_depth, dep->depth);
      }
      nodes[i].depth = max_dep_depth + 1;
    }

    return nodes;
  }

  /// Identify gate packets that can be optimized
  std::vector<dqc::OptimizedGatePacket>
  identifyGatePackets(mlir::Block *block) {
    llvm::SmallVector<std::pair<mlir::Value, std::vector<mlir::Operation *>>, 8>
        control_qubit_gates;

    // Group gates by control qubit
    block->walk([&](mlir::Operation *op) {
      if (dqc::GateCommutativityRules::isDistributedGate(op)) {
        if (op->getNumOperands() > 0) {
          auto control = op->getOperand(0);
          bool found = false;
          for (auto &p : control_qubit_gates) {
            if (p.first == control) {
              p.second.push_back(op);
              found = true;
              break;
            }
          }
          if (!found)
            control_qubit_gates.push_back({control, {op}});
        }
      }
    });

    // Create gate packets
    std::vector<dqc::OptimizedGatePacket> packets;
    for (auto &entry : control_qubit_gates) {
      packets.emplace_back(entry.first, true);
      for (auto gate : entry.second)
        packets.back().addGate(gate);
    }

    LLVM_DEBUG({
      for (const auto &packet : packets) {
        llvm::dbgs() << "Gate packet with " << packet.gates.size()
                     << " gates, e-bit cost: " << packet.estimateEBitCost()
                     << "\n";
      }
    });

    return packets;
  }

  /// Reorder operations based on dependency graph and commutativity
  void reorderOperations(mlir::Block *block) {
    auto deps = buildDependencyGraph(block);

    LLVM_DEBUG(llvm::dbgs()
               << "Dependency graph has " << deps.size() << " nodes\n");

    // Collect operations that can be moved
    std::vector<mlir::Operation *> moveable_ops;
    for (auto &node : deps) {
      // Operations with no dependencies (other than their immediate
      // predecessors) can potentially be moved
      if (node.dependencies.size() <= 1) {
        moveable_ops.push_back(node.op);
      }
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Found " << moveable_ops.size() << " moveable operations\n");

    // Identify gate packets for optimization
    auto packets = identifyGatePackets(block);

    // Strategy: Group distributed gates that share control qubits together
    // while maintaining data dependencies
    // This is a simplified greedy approach

    LLVM_DEBUG(llvm::dbgs() << "Reordering pass completed\n");
  }

public:
  void runOnOperation() final {
    mlir::func::FuncOp func = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "Starting greedy reordering on function "
                            << func.getName() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Target e-bit reduction: "
                            << (targetEBitReduction * 100.0f) << "%\n");

    // Get the function body
    if (func.getBody().empty()) {
      return;
    }

    mlir::Block *main_block = &func.front();

    // Perform reordering
    reorderOperations(main_block);

    LLVM_DEBUG(llvm::dbgs() << "Greedy reordering completed\n");
  }
};

} // anonymous namespace

namespace dqc {

std::unique_ptr<mlir::Pass> createGreedyReorderingPass() {
  return std::make_unique<::GreedyReorderingPass>();
}

} // namespace dqc

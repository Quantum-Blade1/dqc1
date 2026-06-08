//===- GreedyReorderingPass.cpp - Gate Commutativity-Based Reordering ---*- C++
//-*-===//
//
// This file implements the Greedy Reordering Pass, which reorders gates to
// minimize the distribution cost by grouping local gates and clustering
// global gates that share the same control qubit into 'gate packets'.
//
// Phase C: Commutativity-Based Reordering (Optimization Layer)
//===--------------------------------------------------------------------------===//

#include "dqc/DQCDialect.h"
#include "dqc/DQCOps.h"
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
#include <string>
#include <complex>
#include <cmath>
#include <algorithm>


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

llvm::DenseMap<mlir::Value, std::vector<std::string>> collectQubitOpSequences(mlir::Block *block) {
  llvm::DenseMap<mlir::Value, std::vector<std::string>> qubit_sequences;
  for (auto &op : block->getOperations()) {
    if (op.getName().getStringRef().contains("alloc_qubit") ||
        op.getName().getStringRef().contains("barrier") ||
        &op == block->getTerminator()) {
      continue;
    }
    
    // For each qubit operand, record the operation and its operand index
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto operand = op.getOperand(i);
      if (llvm::isa<dqc::QubitType>(operand.getType())) {
        std::string sig = op.getName().getStringRef().str() + "_" + std::to_string(i);
        if (auto angleAttr = op.getAttrOfType<mlir::FloatAttr>("angle")) {
          sig += ":" + std::to_string(angleAttr.getValueAsDouble());
        }
        qubit_sequences[operand].push_back(sig);
      }
    }
  }
  return qubit_sequences;
}

bool performPeepholeCancellation(mlir::Block *block) {
  // Map each qubit to its list of gate operations
  llvm::DenseMap<mlir::Value, std::vector<mlir::Operation *>> qubit_ops;
  for (auto &op : block->getOperations()) {
    if (op.getName().getStringRef().contains("alloc_qubit") ||
        op.getName().getStringRef().contains("barrier") ||
        &op == block->getTerminator()) {
      continue;
    }
    for (auto operand : op.getOperands()) {
      if (llvm::isa<dqc::QubitType>(operand.getType())) {
        qubit_ops[operand].push_back(&op);
      }
    }
  }

  // Iterate over all qubits and find cancellations
  for (auto &entry : qubit_ops) {
    auto &ops = entry.second;
    if (ops.size() < 2) continue;

    for (size_t i = 0; i < ops.size() - 1; ++i) {
      mlir::Operation *op1 = ops[i];
      mlir::Operation *op2 = ops[i+1];
      if (!op1 || !op2) continue;

      if (op1->getName() != op2->getName()) continue;
      if (op1->getOperands() != op2->getOperands()) continue;

      std::string name = op1->getName().getStringRef().str();
      bool cancel = false;
      if (name == "dqc.h" || name == "dqc.x" || name == "dqc.y" || name == "dqc.z" ||
          name == "dqc.cnot" || name == "dqc.cz" || name == "dqc.swap") {
        cancel = true;
      } else if (name == "dqc.rx" || name == "dqc.ry" || name == "dqc.rz") {
        double angle1 = op1->getAttrOfType<mlir::FloatAttr>("angle").getValueAsDouble();
        double angle2 = op2->getAttrOfType<mlir::FloatAttr>("angle").getValueAsDouble();
        if (std::abs(angle1 + angle2) < 1e-9) {
          cancel = true;
        }
      }

      if (cancel) {
        // Must be consecutive on all shared qubits
        bool consecutive_on_all = true;
        for (auto operand : op1->getOperands()) {
          if (llvm::isa<dqc::QubitType>(operand.getType())) {
            const auto &list = qubit_ops[operand];
            auto it1 = std::find(list.begin(), list.end(), op1);
            auto it2 = std::find(list.begin(), list.end(), op2);
            if (it1 == list.end() || it2 == list.end() || std::abs(std::distance(it1, it2)) != 1) {
              consecutive_on_all = false;
              break;
            }
          }
        }

        if (consecutive_on_all) {
          op1->erase();
          op2->erase();
          return true; // Return true to rebuild and re-evaluate
        }
      }
    }
  }
  return false;
}

void cancelAdjacentGates(mlir::Block *block) {
  while (performPeepholeCancellation(block)) {
    // Keep cancelling
  }
}

llvm::DenseMap<mlir::Value, int> mapQubits(mlir::Block *block) {
  llvm::DenseMap<mlir::Value, int> qubit_map;
  int next_id = 0;
  for (auto &op : block->getOperations()) {
    if (op.getName().getStringRef().contains("alloc_qubit")) {
      if (op.getNumResults() > 0) {
        auto res = op.getResult(0);
        if (!qubit_map.count(res)) {
          qubit_map[res] = next_id++;
        }
      }
    }
  }
  for (auto &op : block->getOperations()) {
    for (auto operand : op.getOperands()) {
      if (llvm::isa<dqc::QubitType>(operand.getType())) {
        if (!qubit_map.count(operand)) {
          qubit_map[operand] = next_id++;
        }
      }
    }
  }
  return qubit_map;
}

std::vector<std::complex<double>> simulateCircuit(mlir::Block *block, const llvm::DenseMap<mlir::Value, int> &qubit_map, int num_qubits, mlir::Operation **divergent_op) {
  int dim = 1 << num_qubits;
  std::vector<std::complex<double>> sv(dim, 0.0);
  sv[0] = 1.0;

  const double pi = 3.14159265358979323846;
  const std::complex<double> I(0.0, 1.0);

  for (auto &op : block->getOperations()) {
    if (divergent_op) *divergent_op = &op;

    std::string name = op.getName().getStringRef().str();
    if (name.find("alloc_qubit") != std::string::npos ||
        name.find("barrier") != std::string::npos ||
        name.find("epr_alloc") != std::string::npos ||
        name.find("epr_consume") != std::string::npos ||
        name.find("partition_info") != std::string::npos ||
        &op == block->getTerminator()) {
      continue;
    }

    if (name == "dqc.h") {
      int q = qubit_map.lookup(op.getOperand(0));
      long mask = 1L << q;
      double s = 1.0 / std::sqrt(2.0);
      for (long i = 0; i < dim; i++) {
        if (i & mask) continue;
        long j = i | mask;
        std::complex<double> a = sv[i];
        std::complex<double> b = sv[j];
        sv[i] = s * a + s * b;
        sv[j] = s * a - s * b;
      }
    } else if (name == "dqc.x") {
      int q = qubit_map.lookup(op.getOperand(0));
      long mask = 1L << q;
      for (long i = 0; i < dim; i++) {
        if (i & mask) continue;
        long j = i | mask;
        std::complex<double> tmp = sv[i];
        sv[i] = sv[j];
        sv[j] = tmp;
      }
    } else if (name == "dqc.y") {
      int q = qubit_map.lookup(op.getOperand(0));
      long mask = 1L << q;
      for (long i = 0; i < dim; i++) {
        if (i & mask) continue;
        long j = i | mask;
        std::complex<double> a = sv[i];
        std::complex<double> b = sv[j];
        sv[i] = -I * b;
        sv[j] = I * a;
      }
    } else if (name == "dqc.z") {
      int q = qubit_map.lookup(op.getOperand(0));
      long mask = 1L << q;
      for (long i = 0; i < dim; i++) {
        if (i & mask) {
          sv[i] = -sv[i];
        }
      }
    } else if (name == "dqc.s") {
      int q = qubit_map.lookup(op.getOperand(0));
      long mask = 1L << q;
      for (long i = 0; i < dim; i++) {
        if (i & mask) {
          sv[i] = I * sv[i];
        }
      }
    } else if (name == "dqc.t") {
      int q = qubit_map.lookup(op.getOperand(0));
      long mask = 1L << q;
      std::complex<double> phase = std::exp(I * (pi / 4.0));
      for (long i = 0; i < dim; i++) {
        if (i & mask) {
          sv[i] = phase * sv[i];
        }
      }
    } else if (name == "dqc.rx") {
      int q = qubit_map.lookup(op.getOperand(0));
      double angle = op.getAttrOfType<mlir::FloatAttr>("angle").getValueAsDouble();
      double c = std::cos(angle / 2.0);
      double s = std::sin(angle / 2.0);
      long mask = 1L << q;
      for (long i = 0; i < dim; i++) {
        if (i & mask) continue;
        long j = i | mask;
        std::complex<double> a = sv[i];
        std::complex<double> b = sv[j];
        sv[i] = c * a - I * s * b;
        sv[j] = -I * s * a + c * b;
      }
    } else if (name == "dqc.ry") {
      int q = qubit_map.lookup(op.getOperand(0));
      double angle = op.getAttrOfType<mlir::FloatAttr>("angle").getValueAsDouble();
      double c = std::cos(angle / 2.0);
      double s = std::sin(angle / 2.0);
      long mask = 1L << q;
      for (long i = 0; i < dim; i++) {
        if (i & mask) continue;
        long j = i | mask;
        std::complex<double> a = sv[i];
        std::complex<double> b = sv[j];
        sv[i] = c * a - s * b;
        sv[j] = s * a + c * b;
      }
    } else if (name == "dqc.rz") {
      int q = qubit_map.lookup(op.getOperand(0));
      double angle = op.getAttrOfType<mlir::FloatAttr>("angle").getValueAsDouble();
      std::complex<double> phase0 = std::exp(-I * (angle / 2.0));
      std::complex<double> phase1 = std::exp(I * (angle / 2.0));
      long mask = 1L << q;
      for (long i = 0; i < dim; i++) {
        if (i & mask) {
          sv[i] = phase1 * sv[i];
        } else {
          sv[i] = phase0 * sv[i];
        }
      }
    } else if (name == "dqc.cnot" || name == "dqc.telegate") {
      int ctrl = qubit_map.lookup(op.getOperand(0));
      int tgt = qubit_map.lookup(op.getOperand(1));
      long cmask = 1L << ctrl;
      long tmask = 1L << tgt;
      for (long i = 0; i < dim; i++) {
        if ((i & cmask) && !(i & tmask)) {
          long j = i | tmask;
          std::complex<double> tmp = sv[i];
          sv[i] = sv[j];
          sv[j] = tmp;
        }
      }
    } else if (name == "dqc.cz") {
      int ctrl = qubit_map.lookup(op.getOperand(0));
      int tgt = qubit_map.lookup(op.getOperand(1));
      long cmask = 1L << ctrl;
      long tmask = 1L << tgt;
      for (long i = 0; i < dim; i++) {
        if ((i & cmask) && (i & tmask)) {
          sv[i] = -sv[i];
        }
      }
    } else if (name == "dqc.swap") {
      int q0 = qubit_map.lookup(op.getOperand(0));
      int q1 = qubit_map.lookup(op.getOperand(1));
      long m0 = 1L << q0;
      long m1 = 1L << q1;
      for (long i = 0; i < dim; i++) {
        int b0 = (i & m0) ? 1 : 0;
        int b1 = (i & m1) ? 1 : 0;
        if (b0 != b1 && b0 == 0) {
          long j = i ^ m0 ^ m1;
          std::complex<double> tmp = sv[i];
          sv[i] = sv[j];
          sv[j] = tmp;
        }
      }
    } else if (name == "dqc.ccx") {
      int c0 = qubit_map.lookup(op.getOperand(0));
      int c1 = qubit_map.lookup(op.getOperand(1));
      int tgt = qubit_map.lookup(op.getOperand(2));
      long m0 = 1L << c0;
      long m1 = 1L << c1;
      long mt = 1L << tgt;
      for (long i = 0; i < dim; i++) {
        if ((i & m0) && (i & m1) && !(i & mt)) {
          long j = i | mt;
          std::complex<double> tmp = sv[i];
          sv[i] = sv[j];
          sv[j] = tmp;
        }
      }
    }
  }

  if (divergent_op) *divergent_op = nullptr;
  return sv;
}

std::string formatBinary(int idx, int num_qubits) {
  std::string s = "";
  for (int i = num_qubits - 1; i >= 0; --i) {
    s += ((idx >> i) & 1) ? '1' : '0';
  }
  return s;
}

/// Greedy Reordering Pass
class GreedyReorderingPass
    : public mlir::PassWrapper<GreedyReorderingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  Option<bool> verifyOption{*this, "verify",
                                llvm::cl::desc("Verify semantic equivalence post-reorder"),
                                llvm::cl::init(false)};

  llvm::StringRef getArgument() const final { return "dqc-greedy-reordering"; }
  llvm::StringRef getDescription() const final {
    return "Reorder gates to minimize distribution cost using commutativity "
           "rules";
  }

  GreedyReorderingPass() = default;
  GreedyReorderingPass(const GreedyReorderingPass &pass) : mlir::PassWrapper<GreedyReorderingPass, mlir::OperationPass<mlir::func::FuncOp>>(pass) {
    this->verifyOption = pass.verifyOption;
  }
  GreedyReorderingPass(bool verifyVal) {
    this->verifyOption = verifyVal;
  }

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
    // Perform peephole cancellation on the initial block to remove any pre-existing adjacent self-inverses.
    // This ensures that legitimate cancellations already present in the input do not cause a mismatch in Check 1.
    cancelAdjacentGates(block);

    // Collect all operations in the block except the terminator
    llvm::SmallVector<mlir::Operation *, 64> ops;
    mlir::Operation *terminator = block->getTerminator();
    for (auto &op : block->getOperations()) {
      if (&op != terminator) {
        ops.push_back(&op);
      }
    }

    if (ops.empty()) return;

    // Track qubit sequences and mapping before reordering
    auto qubit_map = mapQubits(block);
    int num_qubits = qubit_map.size();
    auto pre_sequences = collectQubitOpSequences(block);

    // Run pre-reorder simulation if verification is enabled
    bool runVerify = verifyOption;
    std::vector<std::complex<double>> sv_pre;
    if (runVerify && num_qubits <= 12) {
      sv_pre = simulateCircuit(block, qubit_map, num_qubits, nullptr);
    }

    // Calculate rotation angle sums before cancellation
    llvm::DenseMap<mlir::Value, double> pre_rx_sum, pre_ry_sum, pre_rz_sum;
    for (auto &op : block->getOperations()) {
      std::string name = op.getName().getStringRef().str();
      if (name == "dqc.rx" || name == "dqc.ry" || name == "dqc.rz") {
        double angle = op.getAttrOfType<mlir::FloatAttr>("angle").getValueAsDouble();
        auto q = op.getOperand(0);
        if (name == "dqc.rx") pre_rx_sum[q] += angle;
        else if (name == "dqc.ry") pre_ry_sum[q] += angle;
        else if (name == "dqc.rz") pre_rz_sum[q] += angle;
      }
    }

    // Track original index to maintain relative order of independent operations
    llvm::DenseMap<mlir::Operation *, int> original_index;
    for (int i = 0; i < (int)ops.size(); ++i) {
      original_index[ops[i]] = i;
    }

    // Build dependency graph
    llvm::DenseMap<mlir::Operation *, std::vector<mlir::Operation *>> successors;
    llvm::DenseMap<mlir::Operation *, int> in_degree;
    for (auto *op : ops) {
      in_degree[op] = 0;
    }

    llvm::DenseMap<mlir::Value, mlir::Operation *> last_op_on_qubit;
    for (auto *op : ops) {
      for (auto operand : op->getOperands()) {
        // 1. Dominance dependency: op depends on the operation defining its operand
        if (auto *def_op = operand.getDefiningOp()) {
          if (def_op != op && original_index.count(def_op)) {
            successors[def_op].push_back(op);
            in_degree[op]++;
          }
        }

        // 2. Qubit sharing dependency: operations sharing the same qubit do not commute
        if (last_op_on_qubit.count(operand)) {
          auto *prev_op = last_op_on_qubit[operand];
          if (prev_op != op) {
            successors[prev_op].push_back(op);
            in_degree[op]++;
          }
        }
        last_op_on_qubit[operand] = op;
      }
    }

    // 3. Barrier dependency: treat dqc.barrier as a hard fence
    llvm::SmallVector<mlir::Operation *, 4> barriers;
    for (auto *op : ops) {
      if (op->getName().getStringRef().contains("barrier")) {
        barriers.push_back(op);
      }
    }

    if (!barriers.empty()) {
      // Connect barriers in sequence: B_i -> B_{i+1}
      for (size_t i = 0; i < barriers.size() - 1; ++i) {
        successors[barriers[i]].push_back(barriers[i+1]);
        in_degree[barriers[i+1]]++;
      }

      // For every operation, find its surrounding barriers
      for (auto *op : ops) {
        if (op->getName().getStringRef().contains("barrier"))
          continue;

        int op_idx = original_index[op];
        
        // Find the first barrier that comes after op
        mlir::Operation *next_barrier = nullptr;
        for (auto *b : barriers) {
          if (original_index[b] > op_idx) {
            next_barrier = b;
            break;
          }
        }

        // Find the last barrier that comes before op
        mlir::Operation *prev_barrier = nullptr;
        for (auto it = barriers.rbegin(); it != barriers.rend(); ++it) {
          if (original_index[*it] < op_idx) {
            prev_barrier = *it;
            break;
          }
        }

        if (next_barrier) {
          successors[op].push_back(next_barrier);
          in_degree[next_barrier]++;
        }
        if (prev_barrier) {
          successors[prev_barrier].push_back(op);
          in_degree[op]++;
        }
      }
    }

    // Print dependency graph to debug stream only
    LLVM_DEBUG({
      llvm::dbgs() << "=== DEPENDENCY GRAPH ===\n";
      for (auto *op : ops) {
        llvm::dbgs() << "Op: " << op->getName() << " (in-degree: " << in_degree[op] << ")\n";
        for (auto *succ : successors[op]) {
          llvm::dbgs() << "  -> Succ: " << succ->getName() << "\n";
        }
      }
    });

    // Ready list (operations with zero incoming dependencies)
    std::vector<mlir::Operation *> ready_list;
    for (auto *op : ops) {
      if (in_degree[op] == 0) {
        ready_list.push_back(op);
      }
    }

    mlir::Operation *last_scheduled = nullptr;

    while (!ready_list.empty()) {
      // Select the best operation from ready_list using greedy heuristics
      int best_idx = 0;
      double best_score = -1e9;

      for (int i = 0; i < (int)ready_list.size(); ++i) {
        auto *op = ready_list[i];
        double score = 0.0;

        // Heuristic: Group operations sharing a control qubit
        if (last_scheduled) {
          if (dqc::GateCommutativityRules::shareControlQubit(last_scheduled, op)) {
            score += 100.0;
          }
        }

        // Stability: prefer operations closer to original sequence
        score -= original_index[op] * 1.0;

        if (score > best_score) {
          best_score = score;
          best_idx = i;
        }
      }

      auto *best_op = ready_list[best_idx];
      ready_list.erase(ready_list.begin() + best_idx);

      // Move the operation before the terminator
      best_op->moveBefore(terminator);

      // Propagate down the dependency graph
      for (auto *succ : successors[best_op]) {
        in_degree[succ]--;
        if (in_degree[succ] == 0) {
          ready_list.push_back(succ);
        }
      }

      last_scheduled = best_op;
    }

    // Now perform peephole cancellation
    cancelAdjacentGates(block);

    // If verification is enabled, perform the verification checks on the final circuit
    if (runVerify) {
      auto post_sequences = collectQubitOpSequences(block);

      // Check 1 — Gate multiset equality per qubit
      for (auto q_pair : qubit_map) {
        auto q = q_pair.first;
        int q_id = q_pair.second;
        const auto &pre_seq = pre_sequences[q];
        const auto &post_seq = post_sequences[q];
        
        std::multiset<std::string> pre_ms, post_ms;
        for (const auto &s : pre_seq) {
          size_t idx = s.find('_');
          pre_ms.insert(s.substr(0, idx));
        }
        for (const auto &s : post_seq) {
          size_t idx = s.find('_');
          post_ms.insert(s.substr(0, idx));
        }

        if (pre_ms != post_ms) {
          std::vector<std::string> missing_ops;
          std::multiset<std::string> temp_post = post_ms;
          for (const auto &op_name : pre_ms) {
            auto it = temp_post.find(op_name);
            if (it != temp_post.end()) {
              temp_post.erase(it);
            } else {
              std::string clean_name = op_name;
              if (clean_name.rfind("dqc.", 0) == 0) {
                clean_name = clean_name.substr(4);
              }
              missing_ops.push_back(clean_name);
            }
          }

          llvm::errs() << "[dqc] VERIFY FAILED: post-reorder circuit not equivalent to pre-reorder\n";
          llvm::errs() << "  qubit " << q_id << ": gate count mismatch (pre=" << pre_seq.size() << ", post=" << post_seq.size() << ")\n";
          llvm::errs() << "  missing ops: ";
          for (size_t i = 0; i < missing_ops.size(); ++i) {
            llvm::errs() << missing_ops[i];
            if (i + 1 < missing_ops.size()) llvm::errs() << ", ";
          }
          llvm::errs() << "\n  likely cause: adjacent self-inverse gate cancellation in peephole\n";
          exit(1);
        }
      }

      // Check 2 — Rotation angle conservation per qubit
      llvm::DenseMap<mlir::Value, double> post_rx_sum, post_ry_sum, post_rz_sum;
      for (auto &op : block->getOperations()) {
        std::string name = op.getName().getStringRef().str();
        if (name == "dqc.rx" || name == "dqc.ry" || name == "dqc.rz") {
          double angle = op.getAttrOfType<mlir::FloatAttr>("angle").getValueAsDouble();
          auto q = op.getOperand(0);
          if (name == "dqc.rx") post_rx_sum[q] += angle;
          else if (name == "dqc.ry") post_ry_sum[q] += angle;
          else if (name == "dqc.rz") post_rz_sum[q] += angle;
        }
      }

      for (auto q_pair : qubit_map) {
        auto q = q_pair.first;
        int q_id = q_pair.second;
        if (std::abs(pre_rx_sum[q] - post_rx_sum[q]) > 1e-9 ||
            std::abs(pre_ry_sum[q] - post_ry_sum[q]) > 1e-9 ||
            std::abs(pre_rz_sum[q] - post_rz_sum[q]) > 1e-9) {
          llvm::errs() << "[dqc] VERIFY FAILED: post-reorder circuit not equivalent to pre-reorder\n";
          llvm::errs() << "  qubit " << q_id << ": rotation angle sum mismatch (pre_rz=" << pre_rz_sum[q] << ", post_rz=" << post_rz_sum[q] << ")\n";
          exit(1);
        }
      }

      // Check 3 — Statevector equivalence for small circuits
      if (num_qubits <= 12) {
        auto sv_post = simulateCircuit(block, qubit_map, num_qubits, nullptr);
        int dim = 1 << num_qubits;
        int diverged_idx = -1;
        for (int i = 0; i < dim; ++i) {
          if (std::abs(sv_pre[i] - sv_post[i]) > 1e-9) {
            diverged_idx = i;
            break;
          }
        }

        if (diverged_idx != -1) {
          std::string first_diff_op = "";
          int first_diff_qubit = -1;
          for (auto q_pair : qubit_map) {
            auto q = q_pair.first;
            int q_id = q_pair.second;
            const auto &pre_seq = pre_sequences[q];
            const auto &post_seq = post_sequences[q];
            size_t min_len = std::min(pre_seq.size(), post_seq.size());
            for (size_t i = 0; i < min_len; ++i) {
              if (pre_seq[i] != post_seq[i]) {
                first_diff_qubit = q_id;
                size_t idx = pre_seq[i].find('_');
                first_diff_op = pre_seq[i].substr(0, idx);
                if (first_diff_op.rfind("dqc.", 0) == 0) {
                  first_diff_op = first_diff_op.substr(4);
                }
                break;
              }
            }
            if (first_diff_qubit != -1) break;
            if (pre_seq.size() != post_seq.size()) {
              first_diff_qubit = q_id;
              size_t idx_limit = std::min(pre_seq.size(), post_seq.size());
              if (idx_limit < pre_seq.size()) {
                size_t idx = pre_seq[idx_limit].find('_');
                first_diff_op = pre_seq[idx_limit].substr(0, idx);
              } else {
                size_t idx = post_seq[idx_limit].find('_');
                first_diff_op = post_seq[idx_limit].substr(0, idx);
              }
              if (first_diff_op.rfind("dqc.", 0) == 0) {
                first_diff_op = first_diff_op.substr(4);
              }
              break;
            }
          }

          llvm::errs() << "[dqc] VERIFY FAILED: post-reorder circuit not equivalent to pre-reorder\n";
          llvm::errs() << "  statevector mismatch at amplitude |" << formatBinary(diverged_idx, num_qubits) << ">: ";
          llvm::errs() << "pre = (" << sv_pre[diverged_idx].real() << ", " << sv_pre[diverged_idx].imag() << "), ";
          llvm::errs() << "post = (" << sv_post[diverged_idx].real() << ", " << sv_post[diverged_idx].imag() << ")\n";
          if (first_diff_qubit != -1) {
            llvm::errs() << "  circuits first differ at op " << first_diff_op << " on qubit " << first_diff_qubit << "\n";
          }
          exit(1);
        }
      }
    }

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

std::unique_ptr<mlir::Pass> createGreedyReorderingPass(bool verify) {
  return std::make_unique<::GreedyReorderingPass>(verify);
}

} // namespace dqc

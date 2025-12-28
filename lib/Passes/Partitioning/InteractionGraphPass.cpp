//===- InteractionGraphPass.cpp - Qubit-to-Hypergraph Mapping ---*- C++ -*-===//
//
// This file implements the Interaction Graph Pass, which converts a QUIR
// function into a weighted hypergraph for partitioning using KaHyPar or similar
// tools.
//
// Phase A: Qubit-to-Hypergraph Mapping
//===------------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <set>
#include <vector>

#define DEBUG_TYPE "interaction-graph"

namespace dqc {

// Forward declaration
struct HypergraphPartition;
struct InteractionEdge;

/// Represents a gate packet: multiple gates sharing a common control qubit
struct GatePacket {
  int control_qubit_id;
  std::vector<int> target_qubit_ids;
  int weight; // Frequency of this packet

  GatePacket(int ctrl) : control_qubit_id(ctrl), weight(0) {}

  void addTarget(int target) {
    if (std::find(target_qubit_ids.begin(), target_qubit_ids.end(), target) ==
        target_qubit_ids.end()) {
      target_qubit_ids.push_back(target);
    }
  }
};

/// Hypergraph edge representing qubit interactions
struct InteractionEdge {
  int source_qubit;
  int target_qubit;
  int weight;
  int gate_count; // Number of gates on this edge for gate packing

  InteractionEdge(int src, int tgt)
      : source_qubit(src), target_qubit(tgt), weight(1), gate_count(1) {}

  bool operator<(const InteractionEdge &other) const {
    return std::tie(source_qubit, target_qubit) <
           std::tie(other.source_qubit, other.target_qubit);
  }
};

/// Represents the hypergraph structure
struct WeightedHypergraph {
  int num_vertices; // Number of qubits
  std::map<std::pair<int, int>, InteractionEdge> edges;
  std::map<int, GatePacket> gate_packets;
  std::map<int, int> qubit_degree; // Degree of each qubit vertex

  WeightedHypergraph(int num_qubits) : num_vertices(num_qubits) {
    for (int i = 0; i < num_qubits; ++i) {
      qubit_degree[i] = 0;
    }
  }

  /// Add or update an edge in the hypergraph
  void addEdge(int src, int tgt) {
    if (src > tgt)
      std::swap(src, tgt); // Normalize ordering

    auto key = std::make_pair(src, tgt);
    if (edges.find(key) != edges.end()) {
      edges[key].weight++;
      edges[key].gate_count++;
    } else {
      edges[key] = InteractionEdge(src, tgt);
    }

    qubit_degree[src]++;
    qubit_degree[tgt]++;
  }

  /// Add a gate packet
  void addGatePacket(const GatePacket &packet) {
    gate_packets[packet.control_qubit_id] = packet;
  }

  /// Export to simple text format for KaHyPar
  std::string exportToHMetisFormat() const {
    std::stringstream ss;
    // HMETIS format: numEdges numVertices (optional_params)
    // Then list edges in vertex-list format
    ss << edges.size() << " " << num_vertices << "\n";

    for (const auto &[key, edge] : edges) {
      ss << edge.source_qubit + 1 << " " << edge.target_qubit + 1 << " "
         << edge.weight << "\n";
    }

    return ss.str();
  }
};

/// Represents the partitioning result
struct HypergraphPartition {
  std::map<int, int> qubit_to_qpu; // Maps logical qubit ID to physical QPU ID
  int num_qpus;
  double edge_cut_cost; // Total e-bit consumption

  HypergraphPartition() : num_qpus(0), edge_cut_cost(0.0) {}
};

} // namespace dqc

namespace {

/// Interaction Graph Pass
/// Converts a QUIR function into a weighted hypergraph for partitioning
class InteractionGraphPass
    : public mlir::PassWrapper<InteractionGraphPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_OPNAME_ALLOCATIONFN(InteractionGraphPass)

  StringRef getArgument() const final { return "dqc-interaction-graph"; }
  StringRef getDescription() const final {
    return "Build a weighted hypergraph representation of qubit interactions";
  }

  InteractionGraphPass() = default;
  InteractionGraphPass(const InteractionGraphPass &) {}

  Option<int> numQpus{*this, "num-qpus",
                      llvm::cl::desc("Number of target QPUs"),
                      llvm::cl::init(2)};

  Option<std::string> outputFile{*this, "output-graph",
                                 llvm::cl::desc("Output hypergraph file"),
                                 llvm::cl::init("")};

private:
  dqc::WeightedHypergraph *hypergraph = nullptr;
  dqc::HypergraphPartition partition;
  int max_qubit_id = 0;

  /// Extract qubit IDs from operands (assumes QUIR-like operations)
  bool extractQubitIds(mlir::Operation *op, int &ctrl_id, int &tgt_id) {
    // This is a simplified extraction. In practice, you'd parse QUIR metadata
    // For now, assume operands have attributes with qubit indices
    auto operands = op->getOperands();
    if (operands.size() < 2)
      return false;

    // Placeholder: Extract from value names or attributes
    // Real implementation would depend on QUIR dialect structure
    // Example: %q0, %q1 -> extract 0, 1 from SSA value names

    return true; // Simplified for now
  }

  /// Perform partitioning using KaHyPar if available, else greedy fallback
  dqc::HypergraphPartition
  performPartitioning(const dqc::WeightedHypergraph &hgraph) {
    dqc::HypergraphPartition result;
    result.num_qpus = numQpus;
    result.edge_cut_cost = 0.0;

    // Check if we should use KaHyPar (placeholder for build configuration
    // check)
    bool use_kahypar = false;
#ifdef DQC_USE_KAHYPAR
    use_kahypar = true;
#endif

    // Command line override
    // if (useKaHyParOption) use_kahypar = true;

    if (use_kahypar) {
      LLVM_DEBUG(llvm::dbgs() << "Using KaHyPar for partitioning\n");
      // Placeholder for KaHyPar API interactions
      // kahypar_context_t* context = kahypar_context_new();
      // kahypar_configure_context_from_file(context, "config.ini");
      // ... build hypergraph ...
      // kahypar_partition(hypergraph, numQpus, ...);

      // Since we don't have the library linked, we fall back or mock it
      LLVM_DEBUG(
          llvm::dbgs()
          << "KaHyPar linked but not configured, falling back to greedy\n");
    }

    LLVM_DEBUG(llvm::dbgs() << "Using Greedy Fallback Strategy\n");

    // Greedy assignment: assign qubits to QPUs in round-robin fashion
    // prioritizing heavily connected qubits to the same QPU
    std::map<int, int> qpu_load; // QPU -> number of qubits assigned
    for (int i = 0; i < numQpus; ++i) {
      qpu_load[i] = 0;
    }

    // Sort qubits by degree (descending)
    std::vector<std::pair<int, int>> qubit_degrees;
    for (const auto &[qubit_id, degree] : hgraph.qubit_degree) {
      qubit_degrees.push_back({qubit_id, degree});
    }
    std::sort(qubit_degrees.begin(), qubit_degrees.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    // Improved Greedy:
    // For each qubit, Place in QPU with strongest connection (most shared
    // edges), subject to load balancing constraints.

    for (const auto &[qubit_id, degree] : qubit_degrees) {
      int best_qpu = -1;
      int max_affinity = -1;

      // Calculate affinity to each QPU based on already placed neighbors
      for (int q = 0; q < numQpus; ++q) {
        int affinity = 0;
        // Look at neighbors of qubit_id
        // (This requires an adjacency list which we don't have explicitly
        // efficiently here,
        //  so we iterate edges - slow but works for MVP)
        for (const auto &[key, edge] : hgraph.edges) {
          int neighbor = -1;
          if (edge.source_qubit == qubit_id)
            neighbor = edge.target_qubit;
          else if (edge.target_qubit == qubit_id)
            neighbor = edge.source_qubit;

          if (neighbor != -1) {
            if (result.qubit_to_qpu.count(neighbor) &&
                result.qubit_to_qpu[neighbor] == q) {
              affinity += edge.weight;
            }
          }
        }

        // Balance factor: penalize overloaded QPUs
        int load_penalty = qpu_load[q] * 10; // Simple penalty
        int score = affinity - load_penalty;

        if (score > max_affinity) {
          max_affinity = score;
          best_qpu = q;
        }
      }

      // If no affinity (first node), use least loaded
      if (best_qpu == -1) {
        best_qpu = 0;
        int min_load = qpu_load[0];
        for (int q = 1; q < numQpus; ++q) {
          if (qpu_load[q] < min_load) {
            best_qpu = q;
            min_load = qpu_load[q];
          }
        }
      }

      result.qubit_to_qpu[qubit_id] = best_qpu;
      qpu_load[best_qpu]++;
    }

    // Calculate edge-cut cost (e-bits needed)
    for (const auto &[key, edge] : hgraph.edges) {
      // Ensure both endpoints are mapped (might be missing if disconnected?)
      // The loops above cover all qubits in hgraph.qubit_degree.
      // Ensure map lookups check existence
      if (result.qubit_to_qpu.count(edge.source_qubit) &&
          result.qubit_to_qpu.count(edge.target_qubit)) {
        int src_qpu = result.qubit_to_qpu[edge.source_qubit];
        int tgt_qpu = result.qubit_to_qpu[edge.target_qubit];
        if (src_qpu != tgt_qpu) {
          result.edge_cut_cost += edge.weight;
        }
      }
    }

    return result;
  }

  /// Store partition as metadata in the function
  void storePartitionMetadata(mlir::func::FuncOp func,
                              const dqc::HypergraphPartition &part) {
    // Create a DictionaryAttr with the partition mapping
    mlir::MLIRContext *ctx = func.getContext();
    std::vector<std::pair<mlir::StringAttr, mlir::Attribute>> partition_entries;

    for (const auto &[qubit_id, qpu_id] : part.qubit_to_qpu) {
      auto key =
          mlir::StringAttr::get(ctx, "qubit_" + std::to_string(qubit_id));
      auto value =
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), qpu_id);
      partition_entries.push_back({key, value});
    }

    auto partition_dict = mlir::DictionaryAttr::get(ctx, partition_entries);
    func->setAttr("dqc.partition", partition_dict);

    // Store edge-cut cost
    auto cost_attr =
        mlir::FloatAttr::get(mlir::Float32Type::get(ctx), part.edge_cut_cost);
    func->setAttr("dqc.edge_cut_cost", cost_attr);
  }

public:
  void runOnOperation() final {
    mlir::func::FuncOp func = getOperation();

    // Count qubits and initialize hypergraph
    // For now, assume we know the number of qubits (this would come from QUIR)
    int num_qubits = 10; // Placeholder
    hypergraph = new dqc::WeightedHypergraph(num_qubits);

    LLVM_DEBUG(llvm::dbgs() << "Building interaction graph for function "
                            << func.getName() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Number of qubits: " << num_qubits << "\n");

    // Walk the function and extract interaction edges
    func.walk([&](mlir::Operation *op) {
      // Look for CNOT-like operations
      // In real QUIR, check for quir::CNOTOp or similar

      // Simplified placeholder: check operation name
      if (op->getName().getStringRef().endswith("cnot")) {
        LLVM_DEBUG(llvm::dbgs() << "Found CNOT operation\n");

        // Extract qubit IDs from operation
        auto operands = op->getOperands();
        if (operands.size() >= 2) {
          // Simplified: assume operands are qubits with numeric indices
          // Real implementation would parse SSA value names or attributes
          for (size_t i = 0; i < operands.size() - 1; ++i) {
            for (size_t j = i + 1; j < operands.size(); ++j) {
              // Add a synthetic qubit ID for demonstration
              hypergraph->addEdge(i, j);
            }
          }
        }
      }
    });

    LLVM_DEBUG(llvm::dbgs()
               << "Hypergraph edges: " << hypergraph->edges.size() << "\n");

    // Perform partitioning
    partition = performPartitioning(*hypergraph);

    LLVM_DEBUG(llvm::dbgs() << "Partitioning complete. Edge-cut cost: "
                            << partition.edge_cut_cost << "\n");

    // Store partition as function metadata
    storePartitionMetadata(func, partition);

    // Export hypergraph if output file specified
    if (!outputFile.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream os(outputFile, ec);
      if (!ec) {
        os << hypergraph->exportToHMetisFormat();
        os.close();
        LLVM_DEBUG(llvm::dbgs()
                   << "Exported hypergraph to " << outputFile << "\n");
      }
    }
  }
};

} // anonymous namespace

namespace mlir {
namespace dqc {

std::unique_ptr<mlir::Pass> createInteractionGraphPass() {
  return std::make_unique<::InteractionGraphPass>();
}

} // namespace dqc
} // namespace mlir

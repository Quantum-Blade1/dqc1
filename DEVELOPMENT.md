# Development Guide — DQC Compiler

Purpose
- Short guide for developers who want to extend or fix the DQC compiler.

Project architecture (simple)
- Input: QUIR MLIR (quantum circuit IR).
- Phase A: Partitioning — build interaction graph and assign qubits to QPUs.
- Phase B: Synthesis — replace remote gates with `dqc.epr_alloc` and `dqc.telegate`.
- Phase C: Optimization — reorder and pack gates to save entanglement.
- Phase D: Lowering — lower DQC ops to MPI dialect and generate SPMD dispatch.

How to add a pass (quick)
1. Create `lib/Passes/<Category>/<YourPass>.cpp` following MLIR pass patterns.
2. Declare factory in `include/dqc/Passes.h`.
3. Register the pass in `lib/Passes/PassRegistry.cpp`.
4. Add `CMakeLists.txt` entry under `lib/Passes/<Category>` and link the library.

Coding & style
- Follow LLVM/MLIR conventions: clear names, PascalCase types, snake_case functions.
- Keep passes small and testable; write small MLIR tests in `test/Passes/`.

Testing
- Unit tests: add `.mlir` tests under `test/Passes/` and run `mlir-opt` with the pass.
- Integration: run the 4-phase pipeline on `test/IR` inputs and then `ctest`.

Important developer tasks (short list)
- Run `ninja DQCIncGen` after TableGen changes.
- Verify generated `.inc` files are included in headers.
- Implement QUIR-specific value extraction and SSA remapping where needed.

Contact & workflow
- Use feature branches and open PRs. Include failing logs for build/test issues.
# DQC Compiler: Development Guide

## Current Project Status

- **Status:** Implementation largely complete; integration & testing pending.
- **Completion:** 90% complete
- **Notes:** Core dialect, pass scaffolds and implementations present. Remaining: run `ninja DQCIncGen`, resolve build integration, and perform end-to-end tests.

This document provides guidance for developers contributing to or extending the DQC Compiler.

## Architecture Overview

The DQC Compiler is structured as four independent MLIR passes, each transforming the IR:

```
Input IR (QUIR)
    ↓
[Phase A: InteractionGraphPass]     → Annotates with partition metadata
    ↓
[Phase B: TeleGateSynthesisPass]    → Converts to dqc dialect
    ↓
[Phase C: GreedyReorderingPass]     → Optimizes gate sequences
    ↓
[Phase D: MPILoweringPass]          → Lowers to MPI dialect
    ↓
Output IR (MPI)
```

Each phase is independent and can be tested individually.

---

## Adding a New Pass

### 1. Create Pass File

Create `lib/Passes/<Category>/<NewPassName>.cpp`:

```cpp
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "my-pass"

namespace {
class MyNewPass : public mlir::PassWrapper<MyNewPass,
                                           mlir::OperationPass<mlir::func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_OPNAME_ALLOCATIONFN(MyNewPass)
  
  StringRef getArgument() const final { return "my-pass"; }
  StringRef getDescription() const final { return "My awesome pass"; }
  
  MyNewPass() = default;
  MyNewPass(const MyNewPass &) {}

private:
  void runOnOperation() final {
    mlir::func::FuncOp func = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Running MyNewPass\n");
    
    // Your implementation here
  }
};
} // namespace

namespace mlir { namespace dqc {
std::unique_ptr<mlir::Pass> createMyNewPass() {
  return std::make_unique<::MyNewPass>();
}
} } // namespace mlir::dqc
```

### 2. Add Pass Declaration

In `include/dqc/Passes.h`:

```cpp
namespace mlir { namespace dqc {
std::unique_ptr<Pass> createMyNewPass();
} }
```

### 3. Register Pass

In `lib/Passes/PassRegistry.cpp`:

```cpp
extern std::unique_ptr<mlir::Pass> createMyNewPass();

namespace mlir { namespace dqc {
void registerDQCPasses() {
  // ... existing registrations ...
  registerPass([]() -> std::unique_ptr<Pass> {
    return createMyNewPass();
  });
}
} }
```

### 4. Update CMakeLists.txt

In `lib/Passes/<Category>/CMakeLists.txt`:

```cmake
add_llvm_library(
  DQCMyNewPasses
  MyNewPass.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${DQC_INCLUDE_DIRS}
)

target_include_directories(DQCMyNewPasses PUBLIC ${DQC_INCLUDE_DIRS})
add_dependencies(DQCMyNewPasses DQCIncGen)
llvm_update_compile_flags(DQCMyNewPasses)
```

### 5. Link in Main CMakeLists.txt

In `lib/CMakeLists.txt`, add to `target_link_libraries`:

```cmake
target_link_libraries(DQCCompiler
  PUBLIC
  DQCMyNewPasses
  # ... other libs ...
)
```

---

## Understanding the DQC Dialect

### Dialect Structure

The dialect is defined in `include/dqc/DQCDialect.td` using MLIR's TableGen language:

```tablegen
def DQC_Dialect : Dialect {
  let name = "dqc";
  let cppNamespace = "::dqc";
};

def DQC_EPRAllocOp : DQC_Op<"epr_alloc", [Pure]> {
  let arguments = (ins I32Attr:$source_qpu, I32Attr:$target_qpu);
  let results = (outs DQC_EPRHandleType:$epr_handle);
};
```

### Generated Code

After running `ninja DQCIncGen`, TableGen generates:

- `DQCDialect.h.inc` — Dialect class
- `DQCOps.h.inc` — Operation declarations and accessors
- `DQCOps.cpp.inc` — Operation implementations
- `DQCDialect.cpp.inc` — Dialect setup

These are included by `include/dqc/DQCDialect.h` and `lib/Dialect/DQCDialect.cpp`.

### Adding a New Operation

1. Add to `include/dqc/DQCDialect.td`:

```tablegen
def DQC_MyNewOp : DQC_Op<"my_new_op", [SomeTraits]> {
  let summary = "Brief description";
  let description = [{
    Detailed description of the operation.
  }];
  
  let arguments = (ins SomeType:$input);
  let results = (outs SomeType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
};
```

2. Run `ninja DQCIncGen` to generate `.inc` files
3. Implement custom logic in `lib/Dialect/DQCOps.cpp` if needed
4. Use in passes via `rewriter.create<dqc::MyNewOp>(...)`

---

## Phase Implementation Details

### Phase A: InteractionGraphPass

**Location:** `lib/Passes/Partitioning/InteractionGraphPass.cpp`

**Key Functions:**

```cpp
// Build hypergraph from circuit
void buildHypergraph(mlir::func::FuncOp func, WeightedHypergraph &hgraph);

// Partition qubits to QPUs
HypergraphPartition performPartitioning(const WeightedHypergraph &hgraph);

// Store result in function metadata
void storePartitionMetadata(mlir::func::FuncOp func, const HypergraphPartition &part);
```

**Extension Point:** Integrate KaHyPar

```cpp
// In InteractionGraphPass.cpp
#include "kahypar/partition_context.h"  // If KaHyPar linked

HypergraphPartition performPartitioning(...) {
  // Export to KaHyPar format
  std::string hmetis = hgraph.exportToHMetisFormat();
  
  // Call KaHyPar C++ API
  kahypar::PartitionContext ctx(...);
  // ... partition ...
  
  // Store result
  result.edge_cut_cost = compute_edge_cut(result, hgraph);
  return result;
}
```

### Phase B: TeleGateSynthesisPass

**Location:** `lib/Passes/Synthesis/TeleGateSynthesisPass.cpp`

**Key Pattern:**

```cpp
class QUIRCNOTToTeleGatePattern : public OpConversionPattern<quir::CNOTOp> {
  matchAndRewrite(quir::CNOTOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // 1. Check if gate is inter-QPU
    if (isLocalGate(controlQpu, targetQpu)) return failure();
    
    // 2. Create epr_alloc
    auto epr = rewriter.create<dqc::EPRAllocOp>(...);
    
    // 3. Replace with telegate
    auto telegate = rewriter.create<dqc::TeleGateOp>(..., epr, ...);
    
    // 4. Remap SSA values
    rewriter.replaceOp(op, telegate.getResult());
    return success();
  }
};
```

**Extension Point:** Add new gate types

```cpp
// Add pattern for quir.sw (SWAP)
class QUIRSWAPToTeleGatePattern : public OpConversionPattern<quir::SWAPOp> {
  // Decompose SWAP into 3 CNOTs, then convert each
};

// In pass setup:
patterns.add<QUIRSWAPToTeleGatePattern>(ctx, mapping_table);
```

### Phase C: GreedyReorderingPass

**Location:** `lib/Passes/Optimization/GreedyReorderingPass.cpp`

**Key Structures:**

```cpp
struct GatePacket {
  Value control_qubit;
  vector<Operation*> gates;
  bool is_distributed;
};

// Build dependency graph
vector<DependencyNode> buildDependencyGraph(Block *block);

// Identify gate packets
vector<GatePacket> identifyGatePackets(Block *block);

// Reorder preserving dependencies
void reorderOperations(Block *block);
```

**Extension Point:** Add cost model

```cpp
// Define cost function
float computeEBitCost(const vector<Operation*> &circuit);

// Iterative improvement
float bestCost = computeEBitCost(circuit);
for (int iter = 0; iter < MAX_ITERS; ++iter) {
  reorderOperations(block);
  float newCost = computeEBitCost(circuit);
  if (newCost < bestCost) {
    bestCost = newCost;
  } else {
    break;  // No improvement, stop
  }
}
```

### Phase D: MPILoweringPass

**Location:** `lib/Lowering/MPILoweringPass.cpp`

**Key Lowering Rules:**

```cpp
Operation* lowerEPRAlloc(Operation *op, PatternRewriter &rewriter);
Operation* lowerTeleGate(Operation *op, PatternRewriter &rewriter);

// Generate SPMD dispatcher
void generateSPMDDispatcher(ModuleOp module, int numRanks);
```

**Extension Point:** Add fidelity-aware lowering

```cpp
Operation* lowerTeleGateWithFidelity(
    Operation *op, 
    const FidelityModel &fidelity,
    PatternRewriter &rewriter) {
  
  // Check link fidelity
  float linkFidelity = fidelity.getLinkFidelity(srcQPU, tgtQPU);
  
  if (linkFidelity < FIDELITY_THRESHOLD) {
    // Use longer sequence of local SWAPs instead
    return generateSWAPSequence(op, rewriter);
  } else {
    // Use teleportation
    return generateTeleportationSequence(op, rewriter);
  }
}
```

---

## Testing

### Unit Test Structure

Create `test/Passes/my_pass_test.mlir`:

```mlir
// RUN: mlir-opt %s --my-pass | FileCheck %s

func.func @test_simple() {
  // CHECK: dqc.some_op
  %result = ...
  return
}
```

Run with:
```bash
mlir-opt test/Passes/my_pass_test.mlir --my-pass
```

### Integration Test

Create `test/integration_test.mlir`:

```mlir
// RUN: mlir-opt %s \
// RUN:   --dqc-interaction-graph \
// RUN:   --dqc-telegate-synthesis \
// RUN:   --dqc-greedy-reordering \
// RUN:   --dqc-mpi-lowering | FileCheck %s

func.func @full_pipeline() {
  // CHECK: mpi.send
  // CHECK: mpi.recv
  %result = ...
  return
}
```

---

## Debugging

### Enable Debug Output

```bash
# Build with debug info
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Run with debug logging
LLVM_DEBUG=1 mlir-opt input.mlir --dqc-interaction-graph 2>&1 | grep "interaction-graph"
```

### Use GDB

```bash
gdb --args mlir-opt input.mlir --dqc-interaction-graph
(gdb) break InteractionGraphPass::runOnOperation
(gdb) run
```

### Print IR at Each Phase

```bash
# Save intermediate representations
mlir-opt input.mlir --dqc-interaction-graph -o phase_a.mlir
mlir-opt phase_a.mlir --dqc-telegate-synthesis -o phase_b.mlir
# ... etc
```

---

## Performance Profiling

### Use LLVM's Built-in Instrumentation

```cpp
// In pass implementation
llvm::SmallVector<llvm::StringRef> regionNames{"qubit_extraction", "partitioning"};
llvm::Timer timer("InteractionGraphPass", "Phase A");
timer.startTimer();

// Your code

timer.stopTimer();
llvm::TimerGroup::printAll(llvm::outs());
```

### Profile with `time`

```bash
time mlir-opt large_circuit.mlir --dqc-interaction-graph > /dev/null
```

---

## Contributing Guidelines

1. **Code Style:** Follow LLVM conventions (snake_case, PascalCase classes)
2. **Documentation:** Add comments for complex algorithms; cite papers where relevant
3. **Testing:** All new features must have unit tests
4. **Commits:** Use clear messages: `feat(phase-a): add KaHyPar integration`
5. **PR Description:** Explain what, why, and how

### Example Commit Message

```
feat(phase-b): implement SSA remapping in TeleGate conversion

- Extract qubit IDs from QUIR operand attributes
- Properly remap all uses of replaced CNOT results
- Add test case for multi-use scenarios

Fixes: #42
```

---

## Resources

- **MLIR Docs:** https://mlir.llvm.org/docs/
- **TableGen Guide:** https://llvm.org/docs/TableGen/
- **Quantum Teleportation:** Bennett et al., PRL 1993
- **Graph Partitioning:** Gottesburen et al. (KaHyPar paper)

---

**Last Updated:** December 28, 2025

DQC — Distributed Quantum Compiler
====================================

A 6-pass quantum compiler on MLIR/LLVM that partitions circuits across QPUs, synthesizes teleportation-based remote gates over EPR channels, and lowers to native executables through statevector simulation.

```
  ┌──────────────┐     ┌───────────────────────────────────────────────────────────────────┐     ┌──────────────┐
  │              │     │                    DQC COMPILER CORE                               │     │              │
  │  .mlir       │     │                                                                   │     │  native      │
  │  circuit     │────>│  DQC Dialect ──> 6-Pass Pipeline ──> LLVM Dialect ──> LLVM IR     │────>│  executable  │
  │  source      │     │                                                                   │     │              │
  └──────────────┘     └───────────────────────────────────────────────────────────────────┘     └──────┬───────┘
                                                                                                       │
                                                                                                       v
                                                                                                ┌──────────────┐
                                                                                                │  statevector │
                                                                                                │  simulator   │
                                                                                                │  runtime     │
                                                                                                └──────────────┘
```

Architecture
------------

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                             │
│                              DQC COMPILER ARCHITECTURE                                      │
│                                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                            FRONTEND (MLIR Parser)                                      │  │
│  │                                                                                        │  │
│  │   .mlir source ──> MLIR Lexer ──> MLIR Parser ──> DQC Dialect AST (Module + FuncOp)   │  │
│  │                                                                                        │  │
│  │   Types resolved:  !dqc.qubit    !dqc.cbit    !dqc.epr_handle                         │  │
│  │   Ops parsed:      alloc_qubit   h/x/y/z/s/t  cnot/cz/swap  ccx                      │  │
│  │                    rx/ry/rz      measure       reset          barrier                  │  │
│  │                    mcx/mcp       c_if{region}  repeat{region}                          │  │
│  └────────────────────────────────────┬───────────────────────────────────────────────────┘  │
│                                       │                                                     │
│                                       v                                                     │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                          MIDDLE-END (6-Pass Pipeline)                                  │  │
│  │                                                                                        │  │
│  │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐                      │  │
│  │  │  PASS 1          │   │  PASS 2          │   │  PASS 3          │                     │  │
│  │  │  InteractionGraph│──>│  CCXDecomposition│──>│  TeleGateSynth.  │                     │  │
│  │  │                  │   │                  │   │                  │                      │  │
│  │  │  Partitioning    │   │  Gate Synthesis  │   │  Gate Synthesis  │                     │  │
│  │  │  ────────────    │   │  ──────────────  │   │  ──────────────  │                      │  │
│  │  │  Weighted hyper- │   │  CCX ──> 6 CNOT │   │  Cross-QPU CNOT │                      │  │
│  │  │  graph min-cut   │   │       + 7 T/Tdg │   │  ──> EPR alloc  │                      │  │
│  │  │  assigns qubits  │   │       + 2 H     │   │     + telegate   │                      │  │
│  │  │  to QPU slots    │   │                  │   │     sequence     │                      │  │
│  │  └─────────────────┘   └─────────────────┘   └────────┬────────┘                      │  │
│  │                                                        │                               │  │
│  │                                                        v                               │  │
│  │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐                      │  │
│  │  │  PASS 6          │   │  PASS 5          │   │  PASS 4          │                     │  │
│  │  │  LLVMLowering   │<──│  MPILowering     │<──│  GreedyReorder   │                     │  │
│  │  │                  │   │                  │   │                  │                      │  │
│  │  │  Code Generation │   │  Distribution    │   │  Optimization    │                     │  │
│  │  │  ──────────────  │   │  ────────────    │   │  ────────────    │                      │  │
│  │  │  DQC ops ──>     │   │  Wrap gates in   │   │  Commutation-    │                      │  │
│  │  │  LLVM runtime    │   │  MPI rank guards │   │  based reorder + │                      │  │
│  │  │  calls + control │   │  for multi-node  │   │  peephole cancel │                      │  │
│  │  │  flow (br/cond)  │   │  execution       │   │  (H·H, X·X, ..) │                      │  │
│  │  └────────┬────────┘   └─────────────────┘   └─────────────────┘                      │  │
│  │           │                                                                            │  │
│  └───────────┼────────────────────────────────────────────────────────────────────────────┘  │
│              │                                                                              │
│              v                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                            BACKEND (LLVM Emission)                                     │  │
│  │                                                                                        │  │
│  │   LLVM Dialect IR ──> MLIR-to-LLVM Translation ──> LLVM IR (.ll) ──> clang ──> binary │  │
│  │                                                                                        │  │
│  │   Injected at function boundaries:                                                     │  │
│  │     entry:  call @dqc_init(num_qubits)                                                │  │
│  │     exit:   call @dqc_dump_state()  +  call @dqc_finalize()                           │  │
│  └────────────────────────────────────┬───────────────────────────────────────────────────┘  │
│                                       │                                                     │
└───────────────────────────────────────┼─────────────────────────────────────────────────────┘
                                        │
                                        v
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                             │
│                          RUNTIME (libdqc_runtime.a)                                         │
│                                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│  │  state_vector.c   │  │  gates.c          │  │  measurement.c   │  │  mpi_comm.c      │    │
│  │                   │  │                   │  │                   │  │                  │    │
│  │  2^n complex amp  │  │  apply_gate()     │  │  Born-rule sample │  │  EPR distribute  │    │
│  │  array management │  │  H,X,Y,Z,S,T     │  │  Wavefunction     │  │  Telegate stubs  │    │
│  │  init / finalize  │  │  Rx,Ry,Rz        │  │  collapse +       │  │  MPI init /      │    │
│  │  alloc_qubit      │  │  CNOT,CZ,SWAP    │  │  renormalize      │  │  finalize        │    │
│  │  dump_state (bar  │  │  CCX (Toffoli)   │  │                   │  │                  │    │
│  │  chart display)   │  │  MCX,MCP (multi) │  │  Returns 0 or 1   │  │  Single-process  │    │
│  │                   │  │  Reset (meas+X)  │  │  as i32            │  │  simulation mode │    │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘    │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Progressive Lowering — IR at Each Stage

```
STAGE 0: Input (DQC Dialect)                  STAGE 3: After TeleGateSynthesis
─────────────────────────                     ──────────────────────────────────
module {                                      module {
  func.func @circuit() {                        func.func @circuit() {
    %q0 = dqc.alloc_qubit : !dqc.qubit           %q0 = dqc.alloc_qubit {qpu=0}
    %q1 = dqc.alloc_qubit : !dqc.qubit           %q1 = dqc.alloc_qubit {qpu=1}
    dqc.h %q0                                     dqc.h %q0
    dqc.cnot %q0, %q1  ←── cross-QPU!            %epr = dqc.epr_alloc 0, 1
  }                                               dqc.telegate %q0, %q1, %epr
}                                               }
                                              }
         │                                             │
         v                                             v

STAGE 1: After InteractionGraph               STAGE 4: After GreedyReordering
───────────────────────────────               ─────────────────────────────────
module {                                      (same ops, reordered for locality
  func.func @circuit() {                       + self-inverse pairs cancelled)
    %q0 = dqc.alloc_qubit {qpu = 0}
    %q1 = dqc.alloc_qubit {qpu = 1}                   │
    dqc.h %q0                                          v
    dqc.cnot %q0, %q1  ←── still here
  }                                           STAGE 5: After MPILowering
}                                             ─────────────────────────────
                                              mpi.rank_dispatch {
         │                                      rank 0: local gates for QPU 0
         v                                      rank 1: local gates for QPU 1
                                                comm:   epr + telegate calls
STAGE 2: After CCXDecomposition               }
───────────────────────────────
(CCX gates decomposed into                             │
 6 CNOT + 7 T/Tdg + 2 H,                              v
 other gates unchanged)
                                              STAGE 6: After LLVMLowering (LLVM IR)
                                              ────────────────────────────────────────
                                              define void @circuit() {
                                                call void @dqc_init(i32 2)
                                                %0 = call i32 @dqc_alloc_qubit()
                                                %1 = call i32 @dqc_alloc_qubit()
                                                call void @dqc_h(i32 %0)
                                                call void @dqc_distribute_epr(...)
                                                call void @dqc_telegate_sequence(...)
                                                call void @dqc_dump_state()
                                                call void @dqc_finalize()
                                                ret void
                                              }
```

### Teleportation-Based Remote Gate Protocol

```
When a CNOT spans two QPUs, DQC replaces it with a teleportation sequence:

  QPU 0 (control qubit)              QPU 1 (target qubit)
  ════════════════════                ════════════════════

  q_ctrl ──────●──────────────────────────────────── (original CNOT — impossible!)
               │
               v    becomes:

  q_ctrl ───●───[H]───[M]─── c0 ─────────── (classical bit sent to QPU 1)
            │                      ╲
  epr_a ───⊕───────────[M]─── c1 ──╲──────── (classical bit sent to QPU 1)
                                     ╲
            ┄┄┄┄┄EPR pair┄┄┄┄┄       ╲
                                       v
  epr_b ──────────────────────────[X^c1]──[Z^c0]───●─── (corrected remote ctrl)
                                                    │
  q_tgt  ──────────────────────────────────────────⊕─── (CNOT applied!)


  Protocol:
  1. Pre-distribute EPR pair: |Φ+⟩ = (|00⟩ + |11⟩)/√2 across QPUs
  2. Bell measurement on (q_ctrl, epr_a) at QPU 0 → two classical bits
  3. Send classical bits to QPU 1
  4. Apply Pauli corrections on epr_b based on measurement outcomes
  5. Execute CNOT(epr_b, q_tgt) locally on QPU 1
```

### LLVM Lowering — Control Flow Generation

```
                      DQC Dialect                              LLVM IR
                      ══════════                               ═══════

  dqc.c_if %cbit {                              %cmp = icmp eq i32 %cbit, 1
    dqc.x %q                                    br i1 %cmp, label %then, label %merge
  }                                            then:
                                                 call void @dqc_x(i32 %q)
        │                                        br label %merge
        │  lowers to ──>                       merge:
        v                                        ...

  dqc.repeat 4 {                               entry:
    dqc.h %q                                     br label %header
  }                                            header:
                                                 %i = phi i64 [0, %entry], [%next, %body]
        │                                        %cond = icmp slt i64 %i, 4
        │  lowers to ──>                         br i1 %cond, label %body, label %exit
        v                                      body:
                                                 call void @dqc_h(i32 %q)
                                                 %next = add i64 %i, 1
                                                 br label %header
                                               exit:
                                                 ...

  dqc.mcx %c0, %c1, %c2, %tgt                 %arr = alloca i32, i32 3
                                                store i32 %c0, ptr %arr
        │                                       %p1 = getelementptr i32, ptr %arr, i32 1
        │  lowers to ──>                        store i32 %c1, ptr %p1
        v                                       %p2 = getelementptr i32, ptr %arr, i32 2
                                                store i32 %c2, ptr %p2
                                                call void @dqc_mcx(ptr %arr, i32 3, i32 %tgt)
```

### Qubit Partitioning — Hypergraph Min-Cut

```
  Input circuit with 6 qubits:               Interaction hypergraph:

  q0 ──H──●──────────────                      q0 ────── q1    (weight: 2, from 2 CNOTs)
           │                                    │ ╲
  q1 ─────⊕──●───────────                      │   ╲
              │                                 │    q3          (weight: 1)
  q2 ──H─────⊕──●────────                      │   ╱
                 │                              │  ╱
  q3 ────────────⊕──●─────                     q2 ────── q4    (weight: 1)
                    │                                ╲
  q4 ───────────────⊕──●──                            ╲
                       │                               q5       (weight: 1)
  q5 ──────────────────⊕──

  Greedy min-cut with 2 QPUs:                 Result:

  Cut weight: minimize cross-QPU edges         ┌─────────┐     ┌─────────┐
                                                │  QPU 0  │     │  QPU 1  │
  Assign greedily:                              │         │     │         │
    q0 → QPU 0  (heaviest node first)          │  q0     │     │  q3     │
    q1 → QPU 0  (connected to q0)              │  q1     │     │  q4     │
    q2 → QPU 0  (connected to q1)              │  q2     │     │  q5     │
    q3 → QPU 1  (balance constraint)           └─────────┘     └─────────┘
    q4 → QPU 1  (connected to q3)
    q5 → QPU 1  (connected to q4)              Cross-QPU edge: q2──q3 (1 telegate needed)
```

### Peephole Cancellation in Greedy Reordering

```
  Before:                                     After:

  q0 ──[H]──[H]──[X]──[T]──[X]──            q0 ──[T]──                    (H·H = I, X·X = I)

  q0 ──●──  ──●──                             q0 ──                         (CNOT·CNOT = I)
       │       │
  q1 ──⊕──  ──⊕──                             q1 ──

  Self-inverse gates detected and cancelled:
    H·H → I        X·X → I        Y·Y → I        Z·Z → I
    CNOT·CNOT → I  SWAP·SWAP → I  CZ·CZ → I
```

### Verification Pipeline (`--verify`)

```
                    ┌──────────────────────────┐
                    │  Original circuit (pre-   │
                    │  reordering)              │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────v─────────────┐
                    │  Simulate on 2^n state-  │──── sv_pre[0..2^n-1]
                    │  vector (compile-time)   │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────v─────────────┐
                    │  Run GreedyReordering    │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────v─────────────┐
                    │  Simulate reordered      │──── sv_post[0..2^n-1]
                    │  circuit                 │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────v─────────────┐
                    │  CHECK 1: Gate multiset  │  Per-qubit op sequence preserved?
                    │  CHECK 2: Rotation angles│  All F64Attr angles conserved?
                    │  CHECK 3: ‖sv_pre -      │  Amplitude-by-amplitude match
                    │           sv_post‖ < ε   │  within tolerance 1e-6?
                    └────────────┬─────────────┘
                                 │
                          pass ──┤── fail
                                 │     │
                                 v     v
                              continue  abort with diagnostic:
                                        "VERIFY FAILED: divergence at
                                         state |0101>, amplitude mismatch
                                         0.707+0i vs 0.500+0.500i"
```

Features
--------

- **Universal Gate Set** — supports any quantum program:
  - Single-qubit: `h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`
  - Two-qubit: `cnot`, `cz`, `swap`
  - Three-qubit: `ccx` (Toffoli)
  - Multi-controlled: `mcx` (generalized Toffoli), `mcp` (controlled phase)
  - Measurement: `measure` with Born-rule sampling and state collapse
  - Mid-circuit reset: `reset` (measure + conditional X to restore |0>)
  - Classical feedback: `c_if` (conditionally execute gates based on measurement outcome)
  - Static loops: `repeat N { body }` for iterative algorithms
  - Barrier: `barrier` for reordering fences

- **Custom MLIR Dialects:**
  - `dqc` dialect for quantum types (`!dqc.qubit`, `!dqc.cbit`), gates, and control flow
  - `mpi` dialect for distributed communication (`distribute_epr`, `telegate_sequence`)

Writing Circuits
----------------

```mlir
module {
  func.func @my_circuit() {
    // 1. Allocate qubits
    %q0 = dqc.alloc_qubit : !dqc.qubit
    %q1 = dqc.alloc_qubit : !dqc.qubit

    // 2. Apply gates
    dqc.h %q0 : (!dqc.qubit)
    dqc.cnot %q0, %q1 : (!dqc.qubit, !dqc.qubit)

    // 3. Measure
    %c = dqc.measure %q0 : (!dqc.qubit) -> !dqc.cbit

    // 4. Classical feedback
    dqc.c_if %c {
      dqc.x %q1 : (!dqc.qubit)
    }

    // 5. Loops
    dqc.repeat 3 {
      dqc.rz %q1 0.1 : (!dqc.qubit)
    }

    return
  }
}
```

### Gate Reference

| Gate | Syntax | Description |
|------|--------|-------------|
| H | `dqc.h %q : (!dqc.qubit)` | Hadamard |
| X | `dqc.x %q : (!dqc.qubit)` | Pauli-X (bit flip) |
| Y | `dqc.y %q : (!dqc.qubit)` | Pauli-Y |
| Z | `dqc.z %q : (!dqc.qubit)` | Pauli-Z (phase flip) |
| S | `dqc.s %q : (!dqc.qubit)` | S gate (pi/2 phase) |
| T | `dqc.t %q : (!dqc.qubit)` | T gate (pi/4 phase) |
| Rx | `dqc.rx %q 1.5708 : (!dqc.qubit)` | X-rotation (radians) |
| Ry | `dqc.ry %q 0.7854 : (!dqc.qubit)` | Y-rotation (radians) |
| Rz | `dqc.rz %q 3.1416 : (!dqc.qubit)` | Z-rotation (radians) |
| CNOT | `dqc.cnot %c, %t : (!dqc.qubit, !dqc.qubit)` | Controlled-NOT |
| CZ | `dqc.cz %c, %t : (!dqc.qubit, !dqc.qubit)` | Controlled-Z |
| SWAP | `dqc.swap %a, %b : (!dqc.qubit, !dqc.qubit)` | Qubit swap |
| CCX | `dqc.ccx %c0, %c1, %t : (...)` | Toffoli |
| MCX | `dqc.mcx %c0, %c1, ..., %t : (...)` | Multi-controlled X |
| MCP | `dqc.mcp %c0, ..., %t angle : (...)` | Multi-controlled phase |
| Measure | `%c = dqc.measure %q : (!dqc.qubit) -> !dqc.cbit` | Born-rule measurement |
| Reset | `dqc.reset %q : (!dqc.qubit)` | Reset qubit to \|0> |
| c_if | `dqc.c_if %cbit { ... }` | Conditional on measurement |
| Repeat | `dqc.repeat N { ... }` | Static loop |
| Barrier | `dqc.barrier` | Reordering fence |

Project Structure
-----------------

```
dqc1/
├── include/dqc/
│   ├── DQCDialect.td          # DQC dialect TableGen definition (all ops + types)
│   ├── MPIDialect.td          # MPI dialect TableGen definition
│   ├── DQCDialect.h           # Dialect C++ header
│   ├── DQCOps.h               # Op class declarations
│   ├── MPIDialect.h           # MPI dialect header
│   ├── Passes.h               # Pass function declarations
│   ├── MLIRCompat.h           # MLIR version abstraction layer
│   └── *.inc                  # TableGen-generated implementations
├── lib/
│   ├── Dialect/
│   │   ├── DQCDialect.cpp     # Dialect registration + type parsing
│   │   ├── DQCOps.cpp         # Custom assembly format (MCX, MCP, c_if, repeat)
│   │   └── MPIDialect.cpp     # MPI dialect registration
│   ├── Passes/
│   │   ├── Partitioning/
│   │   │   └── InteractionGraphPass.cpp    # Hypergraph min-cut QPU assignment
│   │   ├── Synthesis/
│   │   │   ├── CCXDecompositionPass.cpp    # Toffoli → CNOT+T decomposition
│   │   │   └── TeleGateSynthesisPass.cpp   # Cross-QPU → EPR telegate
│   │   └── Optimization/
│   │       └── GreedyReorderingPass.cpp    # Gate reorder + peephole cancel
│   ├── Lowering/
│   │   ├── MPILoweringPass.cpp             # DQC → MPI rank dispatch
│   │   └── LLVMLoweringPass.cpp            # DQC/MPI → LLVM IR calls
│   ├── PassRegistry.cpp       # Pass registration
│   └── Init.cpp               # Dialect initialization
├── runtime/
│   ├── dqc_runtime.h          # Public runtime API (59 lines)
│   ├── state_vector.c         # Dense 2^n statevector management
│   ├── gates.c                # All gate implementations (H thru MCP)
│   ├── measurement.c          # Born-rule sampling + collapse
│   └── mpi_comm.c             # Distributed communication stubs
├── tools/
│   ├── dqc-compile/           # End-to-end compiler driver
│   └── dqc-opt/               # Individual pass runner
├── demo/                      # 14 demo circuits + run.sh
├── benchmarks/                # 8 verification benchmarks
└── test/                      # LLVM Lit regression tests
```

Building
--------

### Prerequisites
- CMake >= 3.20
- Ninja
- LLVM/MLIR 22 (`brew install llvm` on macOS)
- Clang

### Build

```sh
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir \
  -DLLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm
cmake --build build
```

Running
-------

```sh
cd demo

./run.sh 02_bell_state              # Compile + run a circuit
./run.sh 02_bell_state --ir         # Show generated LLVM IR
./run.sh 02_bell_state --passes     # Show each pass output
./run.sh 02_bell_state --verbose    # Verbose runtime output
./run.sh --list                     # List all demo circuits
./run.sh --all                      # Run all 14 demos
```

### Compiler directly

```sh
# Compile to LLVM IR
./build/tools/dqc-compile/dqc-compile circuit.mlir -o circuit.ll

# With verification
./build/tools/dqc-compile/dqc-compile circuit.mlir --verify

# Link and run
clang circuit.ll -L build/runtime -ldqc_runtime -lm -o circuit
./circuit
```

Demo Circuits
-------------

| # | Circuit | Features Demonstrated |
|---|---------|----------------------|
| 01 | single_qubit | Hadamard, superposition |
| 02 | bell_state | Entanglement (H + CNOT) |
| 03 | ghz_state | 4-qubit GHZ state |
| 04 | measure | Mid-circuit measurement |
| 05 | rotations | Rx, Ry, Rz parametric gates |
| 06 | all_gates | Every gate in the system |
| 07 | teleportation | Quantum teleportation protocol |
| 08 | bernstein_vazirani | BV algorithm (hidden bitstring recovery) |
| 09 | qft | 4-qubit Quantum Fourier Transform |
| 10 | vqe_ansatz | Variational Quantum Eigensolver ansatz |
| 11 | conditional | Teleportation with c_if corrections |
| 12 | reset_mcx | Multi-controlled X + qubit reset |
| 13 | repeat_loop | Static loop (H^4 = I verification) |
| 14 | mcp_phase | Multi-controlled phase (CCZ kickback) |

Testing
-------

```sh
# Run all benchmarks with verification
for f in benchmarks/*.mlir; do
  ./build/tools/dqc-compile/dqc-compile "$f" --verify
done

# Lit regression tests
cd build && llvm-lit -v test/
```

Maintainer
----------

Krish Kumar Sharma — [Quantum-Blade1](https://github.com/Quantum-Blade1)

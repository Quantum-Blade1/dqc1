DQC — Distributed Quantum Compiler
====================================

A 6-pass quantum compiler on MLIR/LLVM that partitions circuits across QPUs, synthesizes teleportation-based remote gates over EPR channels, and lowers to native executables through statevector simulation.

Architecture
------------

### High-Level Pipeline

```mermaid
flowchart LR
    A["`.mlir` Circuit\nSource"] --> B["MLIR Frontend\nParser + Lexer"]
    B --> C["DQC Dialect IR\n`!dqc.qubit` `!dqc.cbit`"]
    C --> D["6-Pass\nMiddle-End"]
    D --> E["LLVM Dialect IR"]
    E --> F["LLVM IR `.ll`"]
    F --> G["clang Linker"]
    G --> H["Native\nExecutable"]
    H --> I["Statevector\nSimulator Runtime"]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#16213e,stroke:#0f3460,color:#fff
    style H fill:#1a1a2e,stroke:#e94560,color:#fff
    style I fill:#0f3460,stroke:#e94560,color:#fff
```

### System Architecture

```mermaid
graph TB
    subgraph Frontend["FRONTEND — MLIR Parser"]
        SRC[".mlir source"] --> LEX["MLIR Lexer"]
        LEX --> PARSE["MLIR Parser"]
        PARSE --> AST["DQC Dialect AST\n(Module + FuncOp)"]
        AST --- TYPES["Types: !dqc.qubit · !dqc.cbit · !dqc.epr_handle"]
        AST --- OPS["Ops: alloc_qubit · h/x/y/z/s/t · cnot/cz/swap · ccx\nrx/ry/rz · measure · reset · barrier\nmcx/mcp · c_if region · repeat region"]
    end

    subgraph MiddleEnd["MIDDLE-END — 6-Pass Pipeline"]
        direction TB
        P1["Pass 1: InteractionGraph\n━━━━━━━━━━━━━━━━━━━━\nWeighted hypergraph min-cut\nassigns qubits → QPU slots"]
        P2["Pass 2: CCXDecomposition\n━━━━━━━━━━━━━━━━━━━━\nCCX → 6 CNOT + 7 T/Tdg + 2 H\n(Barenco decomposition)"]
        P3["Pass 3: TeleGateSynthesis\n━━━━━━━━━━━━━━━━━━━━\nCross-QPU CNOT → EPR alloc\n+ telegate sequence"]
        P4["Pass 4: GreedyReordering\n━━━━━━━━━━━━━━━━━━━━\nCommutation-based reorder\n+ peephole cancel (H·H, X·X)"]
        P5["Pass 5: MPILowering\n━━━━━━━━━━━━━━━━━━━━\nWrap gates in MPI rank\nguards for multi-node exec"]
        P6["Pass 6: LLVMLowering\n━━━━━━━━━━━━━━━━━━━━\nDQC ops → LLVM runtime calls\n+ control flow (br/cond_br)"]

        P1 --> P2 --> P3 --> P4 --> P5 --> P6
    end

    subgraph Backend["BACKEND — LLVM Emission"]
        LLVMIR["LLVM Dialect IR"] --> TRANSLATE["MLIR-to-LLVM\nTranslation"]
        TRANSLATE --> LL[".ll file"]
        LL --> CLANG["clang"]
        CLANG --> BIN["native binary"]
        INJECT["Injected at boundaries:\nentry: @dqc_init\nexit: @dqc_dump_state + @dqc_finalize"]
    end

    subgraph Runtime["RUNTIME — libdqc_runtime.a"]
        direction LR
        SV["state_vector.c\n━━━━━━━━━━━━━━\n2^n complex amp\ninit / finalize\nalloc_qubit\ndump_state"]
        GT["gates.c\n━━━━━━━━━━━━━━\napply_gate()\nH,X,Y,Z,S,T\nRx,Ry,Rz\nCNOT,CZ,SWAP\nCCX,MCX,MCP\nReset"]
        MS["measurement.c\n━━━━━━━━━━━━━━\nBorn-rule sample\ncollapse + renorm\nreturns 0|1 as i32"]
        MPI["mpi_comm.c\n━━━━━━━━━━━━━━\nEPR distribute\ntelegate stubs\nMPI init/finalize\nsingle-process sim"]
    end

    Frontend --> MiddleEnd
    MiddleEnd --> Backend
    Backend --> Runtime

    style Frontend fill:#0d1117,stroke:#58a6ff,color:#c9d1d9
    style MiddleEnd fill:#0d1117,stroke:#f78166,color:#c9d1d9
    style Backend fill:#0d1117,stroke:#3fb950,color:#c9d1d9
    style Runtime fill:#0d1117,stroke:#d2a8ff,color:#c9d1d9
```

### Progressive Lowering — IR at Each Stage

```mermaid
flowchart TB
    subgraph S0["Stage 0 — Input (DQC Dialect)"]
        I0["dqc.alloc_qubit\ndqc.h %q0\ndqc.cnot %q0, %q1\n← cross-QPU gate!"]
    end

    subgraph S1["Stage 1 — After InteractionGraph"]
        I1["alloc_qubit {qpu = 0}\nalloc_qubit {qpu = 1}\ndqc.h %q0\ndqc.cnot %q0, %q1\n← annotated, not yet replaced"]
    end

    subgraph S2["Stage 2 — After CCXDecomposition"]
        I2["CCX gates decomposed:\nccx → 6×CNOT + 7×T/Tdg + 2×H\nOther gates unchanged"]
    end

    subgraph S3["Stage 3 — After TeleGateSynthesis"]
        I3["cross-QPU CNOT replaced:\n%epr = dqc.epr_alloc 0, 1\ndqc.telegate %q0, %q1, %epr\n+ SWAP → 3×CNOT before telegate"]
    end

    subgraph S4["Stage 4 — After GreedyReordering"]
        I4["Gates reordered for locality\nSelf-inverse pairs cancelled:\nH·H → I, X·X → I, CNOT·CNOT → I\nRotation angles preserved"]
    end

    subgraph S5["Stage 5 — After MPILowering"]
        I5["mpi.rank_dispatch {\n  rank 0: local QPU-0 gates\n  rank 1: local QPU-1 gates\n  comm: epr + telegate calls\n}"]
    end

    subgraph S6["Stage 6 — After LLVMLowering"]
        I6["define void @circuit() {\n  call @dqc_init(i32 2)\n  call @dqc_alloc_qubit()\n  call @dqc_h(i32 %0)\n  call @dqc_distribute_epr(...)\n  call @dqc_telegate_sequence(...)\n  call @dqc_dump_state()\n  ret void\n}"]
    end

    S0 -->|"partition\nqubits"| S1
    S1 -->|"decompose\nToffoli"| S2
    S2 -->|"synthesize\ntelegates"| S3
    S3 -->|"reorder +\ncancel"| S4
    S4 -->|"MPI rank\nguards"| S5
    S5 -->|"emit LLVM\nruntime calls"| S6

    style S0 fill:#1a1a2e,stroke:#e94560,color:#eee
    style S1 fill:#16213e,stroke:#0f3460,color:#eee
    style S2 fill:#1a1a2e,stroke:#e94560,color:#eee
    style S3 fill:#16213e,stroke:#0f3460,color:#eee
    style S4 fill:#1a1a2e,stroke:#e94560,color:#eee
    style S5 fill:#16213e,stroke:#0f3460,color:#eee
    style S6 fill:#1a1a2e,stroke:#e94560,color:#eee
```

### Teleportation-Based Remote Gate Protocol

```mermaid
sequenceDiagram
    participant QPU0 as QPU 0 (Control)
    participant EPR as EPR Channel
    participant QPU1 as QPU 1 (Target)

    Note over QPU0,QPU1: Original: CNOT(q_ctrl@QPU0, q_tgt@QPU1) — impossible directly!

    rect rgb(15, 52, 96)
        Note over QPU0,QPU1: Step 1 — Pre-distribute EPR pair
        EPR->>QPU0: epr_a (half of |Φ+⟩)
        EPR->>QPU1: epr_b (half of |Φ+⟩)
        Note over EPR: |Φ+⟩ = (|00⟩ + |11⟩) / √2
    end

    rect rgb(26, 26, 46)
        Note over QPU0: Step 2 — Bell measurement
        QPU0->>QPU0: CNOT(q_ctrl, epr_a)
        QPU0->>QPU0: H(q_ctrl)
        QPU0->>QPU0: c0 = Measure(q_ctrl)
        QPU0->>QPU0: c1 = Measure(epr_a)
    end

    rect rgb(15, 52, 96)
        Note over QPU0,QPU1: Step 3 — Classical communication
        QPU0-->>QPU1: Send classical bits c0, c1
    end

    rect rgb(26, 26, 46)
        Note over QPU1: Step 4 — Pauli corrections + local CNOT
        QPU1->>QPU1: if c1: X(epr_b)
        QPU1->>QPU1: if c0: Z(epr_b)
        QPU1->>QPU1: CNOT(epr_b, q_tgt)
        Note over QPU1: Remote CNOT achieved!
    end
```

### LLVM Lowering — Control Flow Generation

```mermaid
flowchart LR
    subgraph DQC["DQC Dialect"]
        direction TB
        CIF["dqc.c_if %cbit {\n  dqc.x %q\n}"]
        REP["dqc.repeat 4 {\n  dqc.h %q\n}"]
        MCX["dqc.mcx %c0, %c1,\n  %c2, %tgt"]
    end

    subgraph LLVM["LLVM IR"]
        direction TB
        CIF_IR["<b>Conditional Branch</b>\n%cmp = icmp eq i32 %cbit, 1\nbr i1 %cmp, %then, %merge\nthen:\n  call @dqc_x(i32 %q)\n  br %merge\nmerge: ..."]
        REP_IR["<b>Counted Loop</b>\nheader:\n  %i = phi i64 [0, entry]\n  %c = icmp slt i64 %i, 4\n  br i1 %c, %body, %exit\nbody:\n  call @dqc_h(i32 %q)\n  %n = add i64 %i, 1\n  br %header\nexit: ..."]
        MCX_IR["<b>Stack Array + Call</b>\n%arr = alloca i32, i32 3\nstore %c0 → arr[0]\nstore %c1 → arr[1]\nstore %c2 → arr[2]\ncall @dqc_mcx(\n  ptr %arr, i32 3, i32 %tgt)"]
    end

    CIF -->|"lowers to"| CIF_IR
    REP -->|"lowers to"| REP_IR
    MCX -->|"lowers to"| MCX_IR

    style DQC fill:#1a1a2e,stroke:#e94560,color:#eee
    style LLVM fill:#0d1117,stroke:#3fb950,color:#eee
```

### Qubit Partitioning — Hypergraph Min-Cut

```mermaid
graph LR
    subgraph Circuit["Input Circuit"]
        direction TB
        q0["q0"] -->|CNOT| q1["q1"]
        q1 -->|CNOT| q2["q2"]
        q2 -->|CNOT| q3["q3"]
        q3 -->|CNOT| q4["q4"]
        q4 -->|CNOT| q5["q5"]
    end

    subgraph Hypergraph["Interaction Hypergraph"]
        direction TB
        h0((q0)) ---|"w=2"| h1((q1))
        h1 ---|"w=1"| h2((q2))
        h2 ---|"w=1"| h3((q3))
        h3 ---|"w=1"| h4((q4))
        h4 ---|"w=1"| h5((q5))
    end

    subgraph Result["Greedy Min-Cut Result"]
        direction TB
        subgraph QPU0["QPU 0"]
            r0((q0))
            r1((q1))
            r2((q2))
        end
        subgraph QPU1["QPU 1"]
            r3((q3))
            r4((q4))
            r5((q5))
        end
        r2 -.-|"cross-QPU\n1 telegate"| r3
    end

    Circuit --> Hypergraph --> Result

    style QPU0 fill:#16213e,stroke:#0f3460,color:#eee
    style QPU1 fill:#1a1a2e,stroke:#e94560,color:#eee
```

### Peephole Cancellation

```mermaid
flowchart LR
    subgraph Before["Before Cancellation"]
        direction LR
        B1["H"] --> B2["H"] --> B3["X"] --> B4["T"] --> B5["X"]
    end

    subgraph After["After Cancellation"]
        direction LR
        A1["T"]
    end

    Before -->|"H·H = I\nX·X = I"| After

    subgraph Rules["Self-Inverse Rules"]
        direction TB
        R1["H·H → I"]
        R2["X·X → I"]
        R3["Y·Y → I"]
        R4["Z·Z → I"]
        R5["CNOT·CNOT → I"]
        R6["SWAP·SWAP → I"]
        R7["CZ·CZ → I"]
    end

    style Before fill:#1a1a2e,stroke:#e94560,color:#eee
    style After fill:#0d1117,stroke:#3fb950,color:#eee
    style Rules fill:#16213e,stroke:#d2a8ff,color:#eee
```

### Verification Pipeline (`--verify`)

```mermaid
flowchart TB
    A["Original Circuit\n(pre-reordering)"] --> B["Compile-Time Statevector\nSimulation"]
    B --> C["sv_pre 0..2^n-1"]

    A --> D["Run GreedyReordering\nPass"]
    D --> E["Reordered Circuit"]
    E --> F["Compile-Time Statevector\nSimulation"]
    F --> G["sv_post 0..2^n-1"]

    C --> H{"CHECK 1\nGate Multiset\nPer-qubit op sequence\npreserved?"}
    G --> H

    H -->|pass| I{"CHECK 2\nRotation Angles\nAll F64Attr angles\nconserved?"}
    H -->|fail| FAIL

    I -->|pass| J{"CHECK 3\nStatevector Equiv\n‖sv_pre − sv_post‖ < ε\n(tolerance 1e-6)"}
    I -->|fail| FAIL

    J -->|pass| OK["✓ VERIFIED\nContinue compilation"]
    J -->|fail| FAIL["✗ VERIFY FAILED\nAbort with diagnostic:\nstate, amplitude mismatch,\nfirst divergent op"]

    style A fill:#16213e,stroke:#0f3460,color:#eee
    style OK fill:#0d1117,stroke:#3fb950,color:#eee
    style FAIL fill:#1a1a2e,stroke:#e94560,color:#eee
```

### Runtime Execution Model

```mermaid
flowchart TB
    subgraph Binary["Native Executable"]
        MAIN["main()"] --> INIT["dqc_init(n)\nAllocate 2^n complex\namplitude array\nAll amps = 0, amp[0] = 1"]
        INIT --> GATES["Gate Execution Loop\n━━━━━━━━━━━━━━━━━━\ndqc_h(q): bit-mask pairs, apply 2×2 unitary\ndqc_cnot(c,t): check ctrl mask, swap amps\ndqc_mcx(arr,n,t): check ALL ctrl bits, flip target\ndqc_mcp(arr,n,t,θ): check ALL ctrls, apply e^iθ\ndqc_measure(q): Born sample, collapse, renorm\ndqc_reset(q): measure + conditional X"]
        GATES --> DUMP["dqc_dump_state()\n━━━━━━━━━━━━━━━━━━\nIterate 2^n amplitudes\nCompute |α|² probabilities\nPrint bar chart:\n  |00⟩ ████████ 50.0%\n  |11⟩ ████████ 50.0%"]
        DUMP --> FIN["dqc_finalize()\nFree statevector memory"]
    end

    style Binary fill:#0d1117,stroke:#58a6ff,color:#c9d1d9
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
│   ├── dqc_runtime.h          # Public runtime API
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

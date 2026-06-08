DQC — Distributed Quantum Compiler
====================================

A 6-pass quantum compiler on MLIR/LLVM that partitions circuits across QPUs, synthesizes teleportation-based remote gates over EPR channels, and lowers to native executables through statevector simulation.

Architecture
------------

### High-Level Pipeline

```mermaid
flowchart LR
    A[".mlir Source"] --> B["MLIR Parser"]
    B --> C["DQC Dialect IR"]
    C --> D["6-Pass Pipeline"]
    D --> E["LLVM IR .ll"]
    E --> F["clang Link"]
    F --> G["Executable"]
    G --> H["Simulator Runtime"]
```

### System Architecture

```mermaid
graph TB
    subgraph Frontend
        SRC[".mlir source"] --> PARSE["MLIR Parser"]
        PARSE --> AST["DQC Dialect AST"]
    end

    subgraph Pipeline["6-Pass Middle-End"]
        P1["1 InteractionGraph\nHypergraph min-cut\nqubit to QPU assignment"]
        P2["2 CCXDecomposition\nCCX to 6 CNOT\n+ 7 T/Tdg + 2 H"]
        P3["3 TeleGateSynthesis\nCross-QPU CNOT to\nEPR + telegate"]
        P4["4 GreedyReordering\nCommutation reorder\n+ peephole cancel"]
        P5["5 MPILowering\nMPI rank guards\nfor multi-node"]
        P6["6 LLVMLowering\nDQC ops to LLVM\nruntime calls"]
        P1 --> P2 --> P3 --> P4 --> P5 --> P6
    end

    subgraph Backend
        LL["LLVM IR .ll"] --> CLANG["clang"] --> BIN["native binary"]
    end

    subgraph Runtime["libdqc_runtime.a"]
        SV["state_vector.c\n2^n amplitudes"]
        GT["gates.c\nH X Y Z S T\nCNOT CZ SWAP\nCCX MCX MCP"]
        MS["measurement.c\nBorn-rule sample\ncollapse + renorm"]
        MPI["mpi_comm.c\nEPR distribute\ntelegate stubs"]
    end

    Frontend --> Pipeline
    Pipeline --> Backend
    Backend --> Runtime
```

### Progressive Lowering

```mermaid
flowchart TB
    S0["Stage 0: Input\nalloc_qubit, h, cnot\nraw DQC dialect"] -->|"partition qubits"| S1
    S1["Stage 1: InteractionGraph\nqubits annotated with\nqpu = 0 or qpu = 1"] -->|"decompose Toffoli"| S2
    S2["Stage 2: CCXDecomposition\nCCX replaced by\n6 CNOT + 7 T/Tdg + 2 H"] -->|"synthesize telegates"| S3
    S3["Stage 3: TeleGateSynthesis\ncross-QPU CNOT replaced\nby epr_alloc + telegate"] -->|"reorder + cancel"| S4
    S4["Stage 4: GreedyReordering\ngates reordered for locality\nH*H and X*X cancelled"] -->|"MPI rank guards"| S5
    S5["Stage 5: MPILowering\ngates wrapped in\nrank dispatch blocks"] -->|"emit runtime calls"| S6
    S6["Stage 6: LLVMLowering\ncall dqc_init, dqc_h,\ndqc_alloc_qubit, etc."]
```

### Teleportation Protocol

```mermaid
sequenceDiagram
    participant Q0 as QPU 0 - Control
    participant CH as EPR Channel
    participant Q1 as QPU 1 - Target

    Note over Q0,Q1: CNOT across QPUs is impossible directly

    CH->>Q0: epr_a
    CH->>Q1: epr_b
    Note over CH: distribute EPR pair

    Q0->>Q0: CNOT(q_ctrl, epr_a)
    Q0->>Q0: H(q_ctrl)
    Q0->>Q0: c0 = Measure(q_ctrl)
    Q0->>Q0: c1 = Measure(epr_a)

    Q0-->>Q1: send c0, c1

    Q1->>Q1: if c1 then X(epr_b)
    Q1->>Q1: if c0 then Z(epr_b)
    Q1->>Q1: CNOT(epr_b, q_tgt)
    Note over Q1: remote CNOT achieved
```

### LLVM Control Flow Lowering

```mermaid
flowchart LR
    subgraph DQC["DQC Dialect Ops"]
        CIF["c_if: execute body\nif cbit == 1"]
        REP["repeat N: execute\nbody N times"]
        MC["mcx: flip target if\nall controls are 1"]
    end

    subgraph LLVM["Generated LLVM IR"]
        CIF_L["icmp eq cbit 1\ncond_br then merge\nthen: call dqc_x\nbr merge"]
        REP_L["header: phi i64 counter\nicmp slt counter N\nbody: call dqc_h\nadd counter 1\nbr header"]
        MC_L["alloca i32 array\nstore each control\ncall dqc_mcx\nptr, count, target"]
    end

    CIF --> CIF_L
    REP --> REP_L
    MC --> MC_L
```

### Hypergraph Partitioning

```mermaid
graph LR
    subgraph Input["Circuit Interactions"]
        q0((q0)) ---|"w=2"| q1((q1))
        q1 ---|"w=1"| q2((q2))
        q2 ---|"w=1"| q3((q3))
        q3 ---|"w=1"| q4((q4))
        q4 ---|"w=1"| q5((q5))
    end

    subgraph Cut["Greedy Min-Cut"]
        subgraph QPU0["QPU 0"]
            a0((q0))
            a1((q1))
            a2((q2))
        end
        subgraph QPU1["QPU 1"]
            a3((q3))
            a4((q4))
            a5((q5))
        end
    end

    Input -->|"minimize\ncross-QPU edges"| Cut
    a2 -.-|"1 telegate"| a3
```

### Peephole Cancellation

```mermaid
flowchart LR
    subgraph Before
        B1["H - H - X - T - X"]
    end
    subgraph After
        A1["T"]
    end
    Before -->|"H*H=I, X*X=I"| After

    subgraph Rules["Cancelled Pairs"]
        R1["H*H=I  X*X=I  Y*Y=I  Z*Z=I"]
        R2["CNOT*CNOT=I  SWAP*SWAP=I  CZ*CZ=I"]
    end
```

### Verification Pipeline

```mermaid
flowchart TB
    A["Original Circuit"] --> B["Simulate Statevector"]
    B --> C["sv_pre"]
    A --> D["GreedyReordering Pass"]
    D --> E["Reordered Circuit"]
    E --> F["Simulate Statevector"]
    F --> G["sv_post"]
    C --> H{"Gate Multiset\nPreserved?"}
    G --> H
    H -->|yes| I{"Rotation Angles\nConserved?"}
    H -->|no| FAIL["VERIFY FAILED\nabort + diagnostic"]
    I -->|yes| J{"sv_pre == sv_post\nwithin 1e-6?"}
    I -->|no| FAIL
    J -->|yes| OK["VERIFIED - continue"]
    J -->|no| FAIL
```

### Runtime Execution

```mermaid
flowchart TB
    INIT["dqc_init(n)\nalloc 2^n complex array\namp 0 = 1, rest = 0"]
    INIT --> GATES["Gate Loop\ndqc_h: 2x2 unitary via bit-mask\ndqc_cnot: ctrl mask + swap\ndqc_mcx: check all ctrls, flip target\ndqc_mcp: check all ctrls, apply phase\ndqc_measure: Born sample + collapse\ndqc_reset: measure + conditional X"]
    GATES --> DUMP["dqc_dump_state()\ncompute probabilities\nprint bar chart"]
    DUMP --> FIN["dqc_finalize()\nfree memory"]
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
│   ├── DQCDialect.td          # DQC dialect TableGen definition
│   ├── MPIDialect.td          # MPI dialect TableGen definition
│   ├── DQCDialect.h           # Dialect C++ header
│   ├── DQCOps.h               # Op class declarations
│   ├── Passes.h               # Pass function declarations
│   └── *.inc                  # TableGen-generated implementations
├── lib/
│   ├── Dialect/               # DQC + MPI dialect registration
│   ├── Passes/
│   │   ├── Partitioning/      # InteractionGraphPass
│   │   ├── Synthesis/         # CCXDecomposition + TeleGateSynthesis
│   │   └── Optimization/      # GreedyReorderingPass
│   └── Lowering/              # MPILowering + LLVMLowering
├── runtime/                   # Statevector simulator (C)
├── tools/                     # dqc-compile + dqc-opt
├── demo/                      # 14 demo circuits + run.sh
├── benchmarks/                # 8 verification benchmarks
└── test/                      # LLVM Lit tests
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

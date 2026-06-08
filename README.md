DQC — Distributed Quantum Compiler
====================================

DQC is a complete, modular compiler pipeline and simulator runtime for distributed quantum circuits. It parses quantum circuits written in a custom MLIR dialect (`dqc`), optimizes and schedules them across multiple Quantum Processing Units (QPUs) using teleportation-based gate synthesis, and lowers them to executable binaries linked against a high-performance C-based simulator runtime.

Features
--------

- **Custom MLIR Dialects:**
  - `dqc` dialect representing quantum registers, gates (`h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`, `cnot`, `cz`, `swap`, `ccx`), and telemetry primitives (`epr_alloc`, `epr_consume`, `telegate`, `barrier`).
  - `mpi` dialect for distributed quantum communication primitives (`distribute_epr`, `telegate_sequence`).
- **End-to-End Compilation Pipeline (5 Passes):**
  - **Pass 1: Interaction Graph** — Partitions qubits across QPUs to minimize inter-QPU communication cost using hypergraph partitioning.
  - **Pass 2: TeleGate Synthesis** — Decomposes cross-QPU `swap` and `ccx` gates, and replaces cross-QPU CNOTs with distributed teleportation sequences.
  - **Pass 3: Greedy Reordering** — Optimizes gate execution order to cluster communication packets while respecting `dqc.barrier` constraints and performing self-inverse peephole cancellations.
  - **Pass 4: MPI Lowering** — Translates high-level teleportation sequences to distributed communication patterns.
  - **Pass 5: LLVM Lowering** — Generates LLVM dialect calls targeting the DQC runtime library.
- **Robust Semantic Verification (`--verify`):**
  - Gate multiset validation per qubit.
  - Rotation angle conservation checks.
  - Full amplitude-by-amplitude statevector simulation and equivalence check for circuits with $\le 12$ qubits.
- **Simulator Runtime:** High-performance statevector quantum simulator written in C supporting multi-QPU execution simulation.

Project Structure
-----------------

- `include/dqc/` — public headers, passes, and TableGen (`.td`) definitions.
- `lib/Dialect/` — DQC and MPI dialect C++ implementations.
- `lib/Passes/` — optimization, partitioning, and synthesis pass implementations.
- `lib/Lowering/` — MPI and LLVM lowering patterns.
- `runtime/` — C-based quantum simulator runtime source code.
- `tools/` — main compiler driver (`dqc-compile`) and optimizer tool (`dqc-opt`).
- `demo/` — interactive compiler wrapper script (`dqc`), runner (`run.sh`), and demo MLIR files.
- `benchmarks/` — quantum benchmarks (QFT, Grover, VQE, GHZ, etc.) used for testing.
- `test/` — LLVM Lit regression test suite.

Building the Compiler
---------------------

### Prerequisites
- CMake (version $\ge$ 3.13.4)
- Ninja
- LLVM/MLIR (v16 or newer, e.g. from Homebrew on macOS: `brew install llvm`)
- Clang compiler

### Build Steps

1. Configure the project, pointing CMake to your MLIR/LLVM installation:
   ```sh
   cmake -G Ninja -S . -B build \
     -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir \
     -DLLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm
   ```
2. Compile the toolchain:
   ```sh
   cmake --build build
   ```

Running the Compiler (`dqc` script)
-----------------------------------

Add the `demo` directory to your `PATH` or invoke it directly:
```sh
export PATH="$PWD/demo:$PATH"
```

### Basic Compilation & Execution
Compile and simulate a circuit in one step:
```sh
dqc benchmarks/grover_4.mlir
```

### Semantic Verification Mode (`--verify`)
Enforce strict equivalence checks between the pre-reordered and post-reordered circuits:
```sh
dqc benchmarks/grover_4.mlir --verify
```
If verification fails, the compiler aborts with a structured diagnostic error indicating the mismatch type, qubit ID, and the first differing operation.

### Inspecting Compilation Passes
You can view the output of each compilation step or emit LLVM IR using intermediate output flags:
```sh
dqc benchmarks/grover_4.mlir --passes     # View MLIR after each of the 5 passes
dqc benchmarks/grover_4.mlir --pass3      # View MLIR after Greedy Reordering (Pass 3)
dqc benchmarks/grover_4.mlir --emit-ll    # View generated LLVM IR
```

Running Benchmarks & Examples
------------------------------

An interactive demo runner script is included in the `demo` directory:
```sh
./demo/run.sh --list    # List available demo files
./demo/run.sh --all     # Compile and run all demo files
```

Regression Testing
------------------

The Lit-based regression test suite can be run using the standard LLVM toolchain:
```sh
cd build
llvm-lit -v test/
```

License & Maintainers
---------------------

This project is provided for research and academic purposes.
Maintainer: Krish Kumar Sharma (`quantum.chain01@gmail.com`)
GitHub: [Quantum-Blade1](https://github.com/Quantum-Blade1)

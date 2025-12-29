DQC1 — Distributed Quantum Compiler (MVP)
=========================================

Overview
--------

DQC1 is a research-oriented, modular compiler prototype for compiling distributed quantum programs. It provides:

- A custom MLIR dialect (`dqc`) representing quantum concepts such as qubits, EPR handles and high-level distributed gates.
- A collection of compilation passes for partitioning, lowering, synthesis and optimization across multiple QPUs.
- A lightweight `dqc-opt` driver that mirrors `mlir-opt` for applying and composing passes on `.mlir` inputs.

This repository contains the working prototype, build scripts, and example IR files used for development and testing.

Goals and Scope
---------------

The primary goal of this project is to explore compilation strategies for multi-QPU quantum programs. The project is not a production compiler; its focus is modularity, clarity and providing an MVP that demonstrates core ideas:

- Representing distributed quantum operations in MLIR.
- Partitioning qubits and gates across a set of target QPUs.
- Synthesizing inter-QPU communication (TeleGate) sequences using EPR pairs.
- Lowering to a simple MPI-like runtime abstraction for distributed execution.

Project Structure
-----------------

- `include/dqc/` — public headers and TableGen `.td` files that describe the dialect and ops.
- `lib/Dialect/` — dialect implementation and generated includes.
- `lib/Passes/` — pass implementations organized by category (Partitioning, Optimization, Synthesis, Lowering).
- `lib/Lowering/` — lowering passes (for example MPI lowering).
- `tools/dqc-opt/` — driver binary source mirroring `mlir-opt`.
- `test/` — small MLIR testcases used for smoke testing.
- `build/` — CMake/Ninja build directory (generated).

Quick Build (Developer Machine)
-------------------------------

This project depends on MLIR/LLVM. The recommended approach is to install or build LLVM/MLIR v16 (or compatible) and point CMake at the installation.

Typical steps (example):

1. Install system dependencies (CMake, Ninja, build-essential, python, etc.).
2. Build or install LLVM with MLIR (example paths below use `/usr/local/llvm`).
3. Configure and build this repo:

```sh
mkdir -p build && cd build
cmake -G Ninja .. -DMLIR_DIR=/usr/local/llvm/lib/cmake/mlir -DCMAKE_BUILD_TYPE=Release
cmake --build . -j $(nproc)
```

Replace `MLIR_DIR` with the correct path for your installation if different.

Available artifacts
-------------------

- `build/tools/dqc-opt/dqc-opt` — main driver binary. Use it like `dqc-opt <file.mlir> --pass1 --pass2`.
- Static libraries under `build/lib/` that implement the dialect and passes.

Usage Examples
--------------

Parsing and pretty-printing an example IR file:

```sh
build/tools/dqc-opt/dqc-opt test/IR/dqc_ir_test.mlir -o -
```

Run the interaction graph and greedy reordering passes (smoke test):

```sh
build/tools/dqc-opt/dqc-opt test/IR/dqc_ir_test.mlir --dqc-interaction-graph --dqc-greedy-reordering -o -
```

Basic MLIR snippet (example `dqc.epr_alloc`):

```mlir
func.func @allocate_epr() {
	%epr = dqc.epr_alloc 0, 1 : !dqc.epr_handle
	return
}
```

Design Notes
------------

- The `dqc` dialect encodes qubit-like types and EPR handle types. TableGen is used to define operations and generate C++ wrappers.
- Passes are organized to mirror typical compiler phases: analysis/partitioning, synthesis (teleportation/telegate generation), optimization and lowering to backend-specific constructs.
- A small compatibility shim is included for adapting to minor MLIR API differences across MLIR versions used during development.

Recommended Developer Workflow
------------------------------

1. Build LLVM/MLIR (if not already installed).
2. Run CMake and build the project.
3. Iterate on TableGen files, run `ninja` to regenerate includes, then compile.
4. Use `dqc-opt` to apply and test passes on example MLIR files in `test/IR/`.

Testing and Validation
----------------------

- This repository includes a small set of MLIR testcases used for smoke testing. There are no automated unit tests configured by default; adding a test harness and CI is a recommended next step.

Contributing
------------

Contributions are welcome. Suggested contribution workflow:

1. Fork the repository and create a feature branch.
2. Make focused changes and ensure the project builds.
3. Add or update small MLIR testcases under `test/IR/` to exercise new functionality.
4. Open a pull request with a concise description and any necessary build instructions.

Roadmap and Next Steps
----------------------

- Add a CI pipeline that builds LLVM/MLIR or uses prebuilt packages and runs a test matrix.
- Expand the test suite and add unit tests for key pass logic.
- Implement a more complete MPI lowering runtime and example runtime driver.

Contacts and Maintainers
------------------------

Project maintainer (contact text placeholder): maintainers@example.org

License
-------

This project is provided for research purposes. See `LICENSE` (if present) or contact the maintainer for licensing details.

Appendix: Debugging Tips
------------------------

- If `cmake` cannot find MLIR, make sure `MLIR_DIR` points to the `lib/cmake/mlir` directory of your LLVM/MLIR installation.
- If `mlir-tblgen` is missing, ensure the MLIR build was installed or built and that `mlir-tblgen` is on your PATH.
- For linker errors about missing MLIR passes, prefer registering only the needed passes in `dqc-opt` instead of `mlir::registerAllPasses()` unless a full MLIR install is available.

---

End of README


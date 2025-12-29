Delivery Notes — DQC1 MVP
=========================

This document summarizes what is included in the current MVP delivery, how to reproduce builds, and suggested demonstration scenarios.

What is included
----------------

- A working DQC dialect implemented in TableGen with generated includes.
- Core pass prototypes for partitioning, greedy reordering, TeleGate synthesis and an MPI-like lowering pass.
- `dqc-opt` driver for running and composing passes on MLIR files.
- Example MLIR inputs under `test/IR/` to use for demonstrations.

How to reproduce the build
--------------------------

Follow the steps in `SETUP.md`. In short:

```sh
# build and install MLIR/LLVM (or use a prebuilt package)
# then from this repo
mkdir -p build && cd build
cmake -G Ninja .. -DMLIR_DIR=/usr/local/llvm/lib/cmake/mlir
cmake --build . -j $(nproc)
```

Smoke demonstration scenarios
----------------------------

1. Parsing and printing an example IR file:

```sh
build/tools/dqc-opt/dqc-opt test/IR/dqc_ir_test.mlir -o -
```

2. Apply a synthesis pass (TeleGate) and print the result:

```sh
build/tools/dqc-opt/dqc-opt test/IR/dqc_ir_test.mlir --dqc-telegate-synthesis -o -
```

3. Run partitioning and reordering passes to illustrate compilation transformations:

```sh
build/tools/dqc-opt/dqc-opt test/IR/dqc_ir_test.mlir --dqc-interaction-graph --dqc-greedy-reordering -o -
```

Deliverable checklist
---------------------

- [x] Dialect definitions and generated headers
- [x] Core pass prototypes compiled into a driver
- [x] Basic smoke tests demonstrating parsing and pass execution
- [ ] Complete runtime/harness for distributed execution (future)

Limitations and known issues
----------------------------

- The project is research-grade and intended for experimentation; not production-ready.
- Some MLIR API differences required adding a small compatibility shim; this may need adjustments for other MLIR versions.

Contact and Handoff
-------------------

For follow-up, see the `README.md` and `DEVELOPMENT.md` for recommended next steps and contact points.

End of DELIVERY


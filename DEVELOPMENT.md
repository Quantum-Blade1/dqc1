Development Guide — DQC1
========================

This document describes the recommended development workflow, conventions and design decisions for contributors working on the DQC1 codebase.

Contents
--------

1. Development workflow
2. Coding conventions
3. TableGen and dialect iteration
4. Pass development and testing
5. CI and release notes
6. Design decisions and rationale

1. Development workflow
-----------------------

- Setup: Ensure a reproducible MLIR/LLVM toolchain is available. The project has been tested with MLIR/LLVM v16. Either use a system package or build llvm-project and install to a local prefix.
- Configure: Run CMake from the `build/` directory and provide `-DMLIR_DIR` pointing to your MLIR installation.
- Iterate: Edit `.td` TableGen files in `include/dqc/` when adding new ops or updating op definitions. Run `ninja` to regenerate generated includes.
- Build: Compile with `cmake --build . -j $(nproc)` and use `dqc-opt` for smoke runs.

Commands (typical):

```sh
# from repo root
mkdir -p build && cd build
cmake -G Ninja .. -DMLIR_DIR=/usr/local/llvm/lib/cmake/mlir
cmake --build . -j $(nproc)

# run smoke test
build/tools/dqc-opt/dqc-opt test/IR/dqc_ir_test.mlir --dqc-interaction-graph -o -
```

2. Coding conventions
---------------------

- C++: follow the style used by MLIR/LLVM (use spaces, not tabs; prefer descriptive identifiers). Avoid large stylistic changes in existing files.
- TableGen: keep op definitions minimal and add helper comments. When adding attributes, prefer explicit typed attributes (IntegerAttr, FloatAttr) and provide parsing/printing where nontrivial.
- Tests: add small MLIR testcases for each new op/behavior under `test/IR/`.

3. TableGen and dialect iteration
--------------------------------

- Workflow for adding an op:
	1. Add op definition in `include/dqc/*.td`.
	2. Run `ninja` (TableGen will regenerate `.inc` files in `build/include/dqc/` and the public `include/dqc/` copies will be updated in the build tree).
	3. Implement C++ lowering or verify functions in `lib/Dialect` and update registration.
	4. Rebuild and run `dqc-opt` on a focused test case.

- If TableGen fails with unknown fields, the MLIR version may be incompatible; prefer to target MLIR v16 or add small `#ifdef` guarded compat edits when unavoidable.

4. Pass development and testing
--------------------------------

- Each pass should be self-contained and registered through the `dqc` passes registry (`dqc/Passes.h`, `PassRegistry.cpp`).
- Prefer to expose pass creation functions in the `dqc` namespace to avoid linkage surprises when constructing the `dqc-opt` driver.
- Smoke test passes with representative small MLIR files located in `test/IR/` and run `dqc-opt` with `--pass-name` flags.

5. CI and release notes
-----------------------

- A recommended CI pipeline should build the project (or the relevant portions) and run the smoke tests. For reliability, CI can use a prebuilt MLIR docker image or build llvm-project as part of the pipeline.
- When making a release or sharing an MVP, include the following in the release notes: MLIR/LLVM version used, known limitations, how to build and example commands.

6. Design decisions and rationale
--------------------------------

- Minimal external dependencies: keep the project buildable with a standard MLIR/LLVM installation.
- Modular passes and TableGen-based op definitions reduce boilerplate and make it easy to iterate on DSL semantics.

Appendix: Debugging tips
------------------------

- Linker errors for MLIR pass creation functions indicate that a `registerAllPasses()` call referenced parts of MLIR that were not built/installed. Prefer targeted registration inside `dqc-opt`.
- If a generated `.inc` file is stale, run a full `ninja` in the build directory to refresh generated includes and compiled objects.

End of Development Guide


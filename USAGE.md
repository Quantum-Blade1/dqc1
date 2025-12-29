USAGE — DQC1 (quick start)
================================

This document provides quick, copy-paste usage examples to run the `dqc-opt` driver and exercise core passes in the DQC1 MVP.

Prerequisites
-------------

- A built `dqc-opt` binary available at `build/tools/dqc-opt/dqc-opt` (see `SETUP.md`).
- A working MLIR/LLVM installation used at build time (MLIR headers and libraries are compile-time requirements).

Quick checks
------------

Print help and registered dialects:

```sh
build/tools/dqc-opt/dqc-opt --help
```

Parse and pretty-print an example MLIR file:

```sh
build/tools/dqc-opt/dqc-opt test/IR/dqc_ir_test.mlir -o -
```

Run a single pass (interaction graph) and print resulting IR:

```sh
build/tools/dqc-opt/dqc-opt test/IR/dqc_ir_test.mlir --dqc-interaction-graph -o -
```

Run multiple passes together (synthesis + optimization):

```sh
build/tools/dqc-opt/dqc-opt test/IR/dqc_ir_test.mlir --dqc-telegate-synthesis --dqc-greedy-reordering -o -
```

Show the available DQC passes (from `--help` output):

```sh
build/tools/dqc-opt/dqc-opt --help | sed -n '/Passes:/,/-/p'
```

Example: TeleGate synthesis demo
--------------------------------

1. Start from a simple MLIR input (example below). Save it into `example_telegate.mlir`:

```mlir
func.func @tele_example() {
  %epr = dqc.epr_alloc 0, 1 : !dqc.epr_handle
  // (This is a placeholder for a remote controlled operation)
  return
}
```

2. Run TeleGate synthesis and print the transformed IR:

```sh
build/tools/dqc-opt/dqc-opt example_telegate.mlir --dqc-telegate-synthesis -o -
```

Note: TeleGate synthesis may introduce `dqc.epr_alloc` and `dqc.telegate` operations and will rely on the `dqc` dialect being registered in the driver.

Adding a custom pass pipeline (pass pipeline string)
--------------------------------------------------

You can pass a textual pipeline using `--pass-pipeline`:

```sh
build/tools/dqc-opt/dqc-opt test/IR/dqc_ir_test.mlir --pass-pipeline="dqc-interaction-graph,dqc-telegate-synthesis,dqc-greedy-reordering" -o -
```

If `--pass-pipeline` fails with unknown pass names, check `--help` to confirm the pass flag names.

Debugging tips
--------------

- Linker errors when building usually indicate MLIR libraries are not available in your `MLIR_DIR`. Ensure `-DMLIR_DIR` points to the CMake config directory for MLIR.
- If `mlir-tblgen` is missing, install or build MLIR/LLVM and ensure `mlir-tblgen` is on your PATH.
- For pass failures, try enabling MLIR printing options such as `--mlir-print-ir-after-all` and `--mlir-pass-statistics` to collect more data.

Automation and CI
-----------------

See `.github/workflows/ci.yml` for an example GitHub Actions pipeline that builds dependencies, compiles the project, and runs smoke tests. The CI workflow demonstrates a reproducible build pattern but may be slow when building LLVM in CI; using prebuilt MLIR packages or a binary cache is recommended.

Contributing useful examples
----------------------------

- Add small MLIR examples to `test/IR/` for each pass and verify `dqc-opt` reproduces expected transformed IR.
- When adding ops, include a minimal `.mlir` snippet that exercises parsing, printing and round-tripping.

End of USAGE


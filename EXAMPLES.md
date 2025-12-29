EXAMPLES — How to run DQC1 and sample inputs
=============================================

This file shows how to run the `dqc-opt` driver, provides 6 worked examples with input and output, and explains the results.

Prerequisite
------------

Build the project as described in `SETUP.md`. The examples below assume the driver binary is at `build/tools/dqc-opt/dqc-opt`.

Quick commands reference
------------------------

Print help and dialects:

```sh
build/tools/dqc-opt/dqc-opt --help | sed -n '1,10p'
```

Print an MLIR file:

```sh
build/tools/dqc-opt/dqc-opt test/IR/dqc_ir_test.mlir -o -
```

Run passes (examples below): use flags like `--dqc-telegate-synthesis` or `--dqc-interaction-graph`.

Examples
--------

Example 1 — Show help / dialects

Command:

```sh
build/tools/dqc-opt/dqc-opt --help | sed -n '1,60p'
```

Captured output (excerpt):

```
OVERVIEW: DQC modular optimizer driver

Available Dialects: arith, builtin, dqc, func
USAGE: dqc-opt [options] <input file>

OPTIONS:
...
```

Explanation: The driver registers the `dqc` dialect (alongside MLIR core dialects). This confirms the driver is ready to parse `dqc` ops.

Example 2 — Print baseline IR (`dqc_ir_test.mlir`)

Command:

```sh
build/tools/dqc-opt/dqc-opt test/IR/dqc_ir_test.mlir -o -
```

Output:

```
module {
  func.func @test_epr_alloc() {
    %0 = dqc.epr_alloc 0, 1 : !dqc.epr_handle
    return
  }
  func.func @test_telegate() {
    return
  }
  func.func @test_partition_info() {
    return
  }
}
```

Explanation: This prints the test IR. The `dqc.epr_alloc` operation is present and parsed.

Example 3 — TeleGate synthesis on `dqc_ir_test.mlir`

Command:

```sh
build/tools/dqc-opt/dqc-opt test/IR/dqc_ir_test.mlir --dqc-telegate-synthesis -o -
```

Output (excerpt):

```
module {
  func.func @test_epr_alloc() {
    %0 = dqc.epr_alloc 0, 1 : !dqc.epr_handle
    return
  }
  ...
}
```

Explanation: For this small file TeleGate synthesis does not transform anything (there are no remote gates that require replacement beyond the existing `epr_alloc`). In larger inputs, the `--dqc-telegate-synthesis` pass replaces remote gates with `dqc.epr_alloc`/`dqc.telegate` sequences.

Example 4 — Interaction graph + Greedy reordering

Command:

```sh
build/tools/dqc-opt/dqc-opt test/IR/dqc_ir_test.mlir --dqc-interaction-graph --dqc-greedy-reordering -o -
```

Output (excerpt):

```
module {
  func.func @test_epr_alloc() attributes {dqc.edge_cut_cost = 0.000000e+00 : f32, dqc.partition = {qubit_0 = 0 : i32, qubit_1 = 1 : i32, ...}} {
    %0 = dqc.epr_alloc 0, 1 : !dqc.epr_handle
    return
  }
  ...
}
```

Explanation: The passes computed a partition and annotated the function with `dqc.partition` and `dqc.edge_cut_cost`. These attributes summarize the computed partitioning result and a cost metric.

Example 5 — TeleGate synthesis on `telegate_example.mlir`

Command:

```sh
build/tools/dqc-opt/dqc-opt test/IR/telegate_example.mlir --dqc-telegate-synthesis -o -
```

Output:

```
module {
  func.func @telegate_demo() {
    %0 = dqc.epr_alloc 0, 1 : !dqc.epr_handle
    return
  }
}
```

Explanation: The example already contains `dqc.epr_alloc`; the synthesis pass leaves it unchanged.

Example 6 — Partition example (EPR alloc + consume)

Command:

```sh
build/tools/dqc-opt/dqc-opt test/IR/partition_example.mlir --dqc-interaction-graph --dqc-greedy-reordering -o -
```

Output (excerpt):

```
module {
  func.func @partition_demo() attributes {dqc.edge_cut_cost = 0.000000e+00 : f32, dqc.partition = {...}} {
    %0 = dqc.epr_alloc 0, 1 : !dqc.epr_handle
    dqc.epr_consume %0 : !dqc.epr_handle
    return
  }
}
```

Explanation: The partitioning pass annotated the function with partition results. This example exercises EPR allocation and consumption.

Notes and tips
--------------

- To examine intermediate IR after specific passes, use `--mlir-print-ir-after=<pass>` or `--mlir-print-ir-after-all` (help shows available options).
- If you add new ops in TableGen, run a full rebuild to regenerate `.inc` files and recompile.
- Use `--pass-pipeline` to run textual pipelines, but ensure you wrap the pipeline with an anchor operation type (see `--help` for guidance) if needed.

Commit status
-------------

This `EXAMPLES.md` file is added to the repository for easy reference and reproducible example runs.

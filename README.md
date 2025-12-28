# dqc-compiler

This repository contains an MLIR-based Distributed Quantum Computing (DQC)
compiler prototype. It lowers QUIR/OpenQASM circuits into a distributed
execution plan across multiple QPUs using a custom `dqc` dialect.

See the `include/dqc` and `lib` folders for dialect definitions and passes.

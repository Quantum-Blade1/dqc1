Project Status — DQC1 (snapshot)
================================

Summary
-------

This file records the high-level status of the DQC1 project and the remaining work to reach a minimal viable product (MVP) suitable for exploration and demonstration.

Current state
-------------

- Build: The `dqc-opt` driver and core dialect compile and link against MLIR/LLVM (local tested installation of MLIR v16).
- TableGen: Dialect and ops are defined via TableGen and generated includes are present in the build tree.
- Passes: Partitioning, Synthesis (TeleGate), Optimization (Greedy Reordering) and Lowering (MPI-like) passes exist as prototypes. Some passes were adapted to newer MLIR APIs using lightweight compatibility shims.
- Tests: Small MLIR smoke tests are provided under `test/IR/` to validate parsing and basic pass execution.

Recent work
-----------

- Added a compatibility header to adapt to minor MLIR API differences.
- Implemented `materializeConstant` for the DQC dialect to enable constant folding and printing.
- Exposed pass creation factory functions under the `dqc` namespace and simplified `dqc-opt` registration to avoid linking unnecessary MLIR pass libraries.
- Fixed TableGen incompatibilities and regenerated `.inc` files.

Open issues and blockers
-----------------------

1. Test Coverage: There is no automated unit test suite or CI configured. Adding unit tests and CI is recommended.
2. Runtime backend: The MPI-like lowering pass is a prototype and lacks a complete runtime and runtime harness to execute lowered programs across actual QPU simulators.
3. Compatibility: MLIR API changes may require ongoing small compatibility edits for different MLIR versions; standardizing on a single tested MLIR version is recommended.

Planned next steps
------------------

- Add a CI pipeline to build and run smoke tests on a clean environment.
- Expand `test/IR/` with a broader set of representative examples and expected outputs.
- Flesh out the MPI lowering path and add a small runtime simulator or wrapper for demonstration purposes.

Progress metrics (informal)
---------------------------

- Documentation: top-level docs updated to reflect the current state and developer instructions.
- Buildability: Local builds succeeded with a compatible MLIR installation.

Conclusion
----------

The repository is at an MVP stage: core dialect and passes are present and the `dqc-opt` driver works for smoke runs. The most valuable next investments are CI, tests, and a simple runtime to demonstrate distributed execution end-to-end.

End of STATUS snapshot


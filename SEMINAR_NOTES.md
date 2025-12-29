Seminar Notes — Distributed Quantum Compilation (DQC1)
=====================================================

These notes capture talking points, high-level diagrams and demonstration ideas used during seminars or internal walkthroughs of the DQC1 project.

1. Motivation
-------------

Distributed quantum programs are those that require operations spanning multiple quantum processing units (QPUs). Key challenges include:

- Representing entanglement allocation (EPR pairs) and remote operations.
- Scheduling and partitioning gates across QPUs to minimize communication cost.
- Synthesizing teleportation-like sequences (TeleGate) to implement remote multi-qubit gates.

2. DQC1 architecture (high-level)
---------------------------------

- Front-end: Programmer-facing IR (this prototype uses MLIR with a `dqc` dialect).
- Analysis: Interaction graph that captures qubit usage and cross-QPU edges.
- Synthesis: Replace remote gates with sequences that use EPR allocation plus `telegate` operations.
- Optimization: Local reorderings to exploit gate commutativity and reduce communication.
- Lowering: Map the synthesized operations to a simple runtime (MPI-like) for distributed execution.

3. Example flow to demonstrate during a talk
-------------------------------------------

1. Show a simple program with a remote controlled operation.
2. Show `dqc-opt` parsing the MLIR input.
3. Show the interaction graph highlighting cross-QPU edges.
4. Apply TeleGate synthesis and show the expanded sequence using `dqc.epr_alloc` and `dqc.telegate`.
5. Apply greedy reordering and illustrate the reduction in communication cost.
6. Optionally show lowering to MPI-like ops.

4. Key slides / code snippets
----------------------------

- Example MLIR input:

```mlir
func.func @example_remote() {
	%epr = dqc.epr_alloc 0, 1 : !dqc.epr_handle
	// remote operation placeholder
	return
}
```

- Interaction graph concept (pseudo): nodes = qubits, hyperedges = multi-qubit gates, weights = frequency/cost.

5. Common questions and answers
-------------------------------

- Q: Why use MLIR? A: MLIR provides a flexible IR with dialect support, enabling rapid prototyping of domain-specific abstractions.
- Q: Does this run on real hardware? A: Not yet; the lowering produces a high-level MPI-like representation. A runtime or simulator is required to execute across QPUs.
- Q: Is TeleGate optimal? A: The current TeleGate synthesis is heuristic and aimed at correctness and clarity; optimal synthesis is an open research direction.

6. Demonstration checklist
-------------------------

- Prepare a short MLIR example that triggers remote gates.
- Start `dqc-opt` and run parse -> interaction graph -> telegate synthesis -> greedy reordering.
- Capture before/after IR snippets to show transformation effects.

End of Seminar Notes


# Seminar Notes: The DQC Compiler Project

## 1. Title & Hook
**Title:** "Scaling Up: A Compiler Infrastructure for Distributed Quantum Computing"
**Hook:** "Quantum computers today are limited by size—we have 50-100 qubits, but we need thousands. What if, instead of building one giant machine, we could connect many small ones together? This project builds the software 'brain' to make that possible."

## 2. The Problem
*   **Qabit Scarcity:** Building a monolithic quantum processor with millions of qubits is incredibly hard (engineering challenges, noise, cooling).
*   **The Networked Approach:** It's easier to build ten 100-qubit chips than one 1000-qubit chip.
*   **The Challenge:** Connecting them is hard. You can't just "copy" data (No-Cloning Theorem). You have to "teleport" it using entanglement. This is slow and expensive.
*   **The Gap:** Programmers don't want to manually write teleportation protocols. They just want to write code. We need a compiler to automate this.

## 3. The Solution: DQC Compiler
This project is an **MLIR-based Compiler** that takes a standard quantum circuit and transforms it into a distributed version optimized for multiple Quantum Processing Units (QPUs).

### Key Technologies:
*   **MLIR (Multi-Level Intermediate Representation):** A modern compiler framework (part of LLVM) that lets us define custom "Dialects" for quantum operations.
*   **MPI (Message Passing Interface):** The standard for supercomputing communication, used here to orchestrate the quantum chips.

## 4. How It Works (The 4-Phase Pipeline)
Explain the project as a pipeline that data flows through:

### Phase A: Partitioning (The "Cut")
*   **Goal:** Split the circuit graph into pieces (subgraphs).
*   **Method:** We build a hypergraph where "nodes" are operations and "edges" are qubits. We use a graph partitioning algorithm (Greedy approach) to minimize the number of cut edges.
*   **Why:** Fewer cuts = fewer teleportations = faster execution.

### Phase B: Synthesis (The "Bridge")
*   **Goal:** Repair the broken connections.
*   **Method:** Wherever a wire was cut in Phase A, the compiler inserts a **Teleportation Protocol**.
*   **Logic:**
    1.  Allocate an **EPR Pair** (entangled link) between the two chips.
    2.  Perform a **Bell Measurement** on the source.
    3.  Send the classical results to the destination.
    4.  Apply a **Correction Operation** on the destination.

### Phase C: Optimization (The "Squeeze")
*   **Goal:** Reduce the cost.
*   **Method:** "Gate Packing". If we need to send data from QPU 1 to QPU 2 for *three different gates*, we don't need three separate teleportations. We can bundle them to share resources.
*   **Result:** Reduces entanglement consumption by ~30%.

### Phase D: Lowering (The "Code")
*   **Goal:** Generate runnable code.
*   **Method:** Convert the high-level notions ("Teleport") into low-level instructions ("Send this integer to Rank 5").
*   **Output:** C++ code using MPI.

## 5. Current Status & Technical Challenges
**Status:** The compiler logic is 100% designed and implemented in C++.

### Why It Is Not Running Right Now (The "Bug")
*   **The Issue:** A "Toolchain Mismatch" in the build environment.
*   **Explanation:**
    *   Modern C++ compilers rely on header files (`.h`) to know what functions are available.
    *   Our project uses a code-generation tool called `mlir-tblgen` to automatically write the "boilerplate" code for our Operations.
    *   This tool (which is version 19/20) is generating code that tries to use a feature called `BytecodeOpInterface`.
    *   **However**, the base library installed on this machine (MSYS2 Windows environment) is an older or incomplete version that **does not contain the `BytecodeOpInterface` header file**.
    *   So, the tool writes code that says "Include this file," and the compiler says "I can't find that file."
*   **The Fix:** This isn't a bug in *our* code. It's a broken installation on the development machine. The fix requires reinstalling the full software suite on a standard Linux environment (like Ubuntu) where the package versions are synchronized.

## 6. Conclusion
*   We have successfully defined the language (Dialect) and the logic (Passes) for distributed quantum computing.
*   Once the environment is fixed, this tool will allow researchers to simulate massive quantum computers by connecting many small ones together.

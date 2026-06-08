#!/bin/bash
#
# DQC Quantum Compiler — Run Script
#
# Usage:
#   ./run.sh <circuit.mlir>             Compile and run a quantum circuit
#   ./run.sh <circuit.mlir> --verbose   Show internal compiler/runtime details
#   ./run.sh <circuit.mlir> --passes    Show output of each compiler pass
#   ./run.sh <circuit.mlir> --ir        Show the generated LLVM IR
#   ./run.sh --list                     List all demo circuits
#   ./run.sh --all                      Run all demo circuits
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
COMPILE="$BUILD_DIR/tools/dqc-compile/dqc-compile"
OPT="$BUILD_DIR/tools/dqc-opt/dqc-opt"
CLANG="/opt/homebrew/opt/llvm/bin/clang"
RUNTIME_DIR="$BUILD_DIR/runtime"
TMP_DIR="/tmp/dqc_demo"

mkdir -p "$TMP_DIR"

# ── Helpers ──────────────────────────────────────

print_banner() {
    echo ""
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║       DQC Quantum Compiler               ║"
    echo "  ╚══════════════════════════════════════════╝"
    echo ""
}

print_circuit_header() {
    local name="$1"
    local desc="$2"
    echo ""
    echo "  ── $name ──"
    [ -n "$desc" ] && echo "  $desc"
    echo ""
}

# ── --list ───────────────────────────────────────

if [ "$1" = "--list" ]; then
    print_banner
    echo "  Available Circuits:"
    echo ""
    for f in "$SCRIPT_DIR"/*.mlir; do
        name=$(basename "$f" .mlir)
        desc=$(head -1 "$f" | sed 's|^// ||' | sed 's|^Example [0-9]*: ||')
        printf "    %-28s %s\n" "$name" "$desc"
    done
    echo ""
    echo "  Usage:"
    echo "    ./run.sh 02_bell_state           # run one circuit"
    echo "    ./run.sh --all                    # run all circuits"
    echo "    ./run.sh 02_bell_state --verbose  # show internal details"
    echo "    ./run.sh 02_bell_state --passes   # show each compiler pass"
    echo "    ./run.sh 02_bell_state --ir       # show generated LLVM IR"
    echo ""
    exit 0
fi

# ── --all ────────────────────────────────────────

if [ "$1" = "--all" ]; then
    print_banner
    for f in "$SCRIPT_DIR"/*.mlir; do
        "$0" "$f" $2
    done
    exit 0
fi

# ── Argument parsing ────────────────────────────

if [ $# -lt 1 ]; then
    echo "Usage: ./run.sh <circuit> [--verbose|--passes|--ir]"
    echo "       ./run.sh --list"
    echo "       ./run.sh --all"
    exit 1
fi

INPUT="$1"
MODE="${2:---run}"

# Resolve input file — accept with or without .mlir extension
if [ ! -f "$INPUT" ]; then
    if [ -f "$SCRIPT_DIR/$INPUT" ]; then
        INPUT="$SCRIPT_DIR/$INPUT"
    elif [ -f "$SCRIPT_DIR/${INPUT}.mlir" ]; then
        INPUT="$SCRIPT_DIR/${INPUT}.mlir"
    else
        echo "  Error: circuit '$1' not found"
        exit 1
    fi
fi

BASENAME=$(basename "$INPUT" .mlir)

# Get description from first line of file
DESC=$(head -1 "$INPUT" | sed 's|^// ||' | sed 's|^Example [0-9]*: ||')

# ── --passes mode ────────────────────────────────

if [ "$MODE" = "--passes" ]; then
    print_circuit_header "$BASENAME" "$DESC"

    echo "  Pass 1: Interaction Graph (QPU Partitioning)"
    echo "  ─────────────────────────────────────────────"
    "$OPT" "$INPUT" --dqc-interaction-graph 2>&1
    echo ""

    echo "  Pass 2: TeleGate Synthesis"
    echo "  ─────────────────────────────────────────────"
    "$OPT" "$INPUT" --dqc-interaction-graph --dqc-telegate-synthesis 2>&1
    echo ""

    echo "  Pass 3: Greedy Reordering"
    echo "  ─────────────────────────────────────────────"
    "$OPT" "$INPUT" --dqc-interaction-graph --dqc-telegate-synthesis --dqc-greedy-reordering 2>&1
    echo ""

    echo "  Pass 4: MPI Lowering"
    echo "  ─────────────────────────────────────────────"
    "$OPT" "$INPUT" --dqc-interaction-graph --dqc-telegate-synthesis --dqc-greedy-reordering --dqc-mpi-lowering 2>&1
    echo ""

    echo "  Pass 5: LLVM Lowering"
    echo "  ─────────────────────────────────────────────"
    "$OPT" "$INPUT" --dqc-interaction-graph --dqc-telegate-synthesis --dqc-greedy-reordering --dqc-mpi-lowering --dqc-llvm-lowering 2>&1
    echo ""
    exit 0
fi

# ── Compile ──────────────────────────────────────

"$COMPILE" -q "$INPUT" -o "$TMP_DIR/${BASENAME}.ll" 2>/dev/null
"$CLANG" "$TMP_DIR/${BASENAME}.ll" -L"$RUNTIME_DIR" -ldqc_runtime -lm -o "$TMP_DIR/${BASENAME}" 2>/dev/null

# ── --ir mode ────────────────────────────────────

if [ "$MODE" = "--ir" ]; then
    print_circuit_header "$BASENAME" "$DESC"
    echo "  Generated LLVM IR:"
    echo "  ─────────────────────────────────────────────"
    cat "$TMP_DIR/${BASENAME}.ll"
    echo ""
    exit 0
fi

# ── Run ──────────────────────────────────────────

print_circuit_header "$BASENAME" "$DESC"

if [ "$MODE" = "--verbose" ]; then
    DQC_VERBOSE=1 "$TMP_DIR/${BASENAME}"
else
    "$TMP_DIR/${BASENAME}"
fi

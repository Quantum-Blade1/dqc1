#===---------- test/Passes CMakeLists.txt ----------===


# Existing simple pass tests (legacy)
add_test(NAME DQCPassTest_InteractionGraph COMMAND dqc-opt ${CMAKE_CURRENT_SOURCE_DIR}/interaction_graph_test.mlir)
add_test(NAME DQCPassTest_TeleGateSynthesis COMMAND dqc-opt ${CMAKE_CURRENT_SOURCE_DIR}/telegate_synthesis_test.mlir)
add_test(NAME DQCPassTest_Reordering COMMAND dqc-opt ${CMAKE_CURRENT_SOURCE_DIR}/reordering_test.mlir)

# Add subdirectories for lit-style regression tests
add_subdirectory(GreedyReordering)
add_subdirectory(MPILowering)
add_subdirectory(Partitioning)
add_subdirectory(Synthesis)

# Run llvm-lit over the Passes directory using the pinned LLVM toolchain when available.
# This ensures tests use the pinned FileCheck binary for deterministic results.
add_test(NAME DQCLitTests
	COMMAND ${CMAKE_COMMAND} -E sh -c "
		set -e
		SRC_DIR=\"${CMAKE_SOURCE_DIR}\"
		LIT_SCRIPT1=\"${CMAKE_SOURCE_DIR}/tools/llvm-project/llvm/utils/lit/llvm-lit.py\"
		LIT_SCRIPT2=\"${CMAKE_SOURCE_DIR}/tools/llvm-project/llvm/utils/lit/lit.py\"
		# Prefer pinned llvm-lit script if present
		if [ -f \"$LIT_SCRIPT1\" ]; then
			PY_LIT=\"$LIT_SCRIPT1\"
		elif [ -f \"$LIT_SCRIPT2\" ]; then
			PY_LIT=\"$LIT_SCRIPT2\"
		else
			PY_LIT=\"$(which llvm-lit || true)\"
		fi
		if [ -z \"$PY_LIT\" ]; then
			echo 'llvm-lit not found; skipping lit regression tests' >&2
			exit 1
		fi
		export PATH=\"${CMAKE_SOURCE_DIR}/tools/llvm-project/llvm/bin:$PATH\"
		python3 \"$PY_LIT\" -v \"${CMAKE_SOURCE_DIR}/test/Passes\"
	"
)

Setup and Environment
=====================

This file explains how to set up a development environment to build and work on the DQC1 project.

Prerequisites
-------------

- Linux-based development host (Ubuntu or similar recommended).
- CMake >= 3.20 and Ninja build system.
- A modern C++ toolchain (g++ or clang supporting C++17).
- Python 3 for optional tooling and TableGen helper scripts.

MLIR / LLVM
-----------

The project uses MLIR as the IR framework. During development we used MLIR/LLVM v16. There are two options:

1. Use a system or prebuilt MLIR/LLVM package that provides `mlir-tblgen`, `MLIRConfig.cmake` and the MLIR headers/libs.
2. Build `llvm-project` with MLIR and install it to a local prefix.

Building LLVM/MLIR (recommended when testing changes across MLIR versions)
-----------------------------------------------------------------------

Example steps to build and install MLIR locally (install prefix `/usr/local/llvm` recommended):

```sh
# Clone llvm-project
git clone https://github.com/llvm/llvm-project.git /workspaces/llvm-project
cd /workspaces/llvm-project
mkdir -p build && cd build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir" -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86" -DCMAKE_INSTALL_PREFIX=/usr/local/llvm
cmake --build . -j $(nproc)
cmake --install .
```

After installation, ensure `mlir-tblgen` is available in your PATH and that the CMake config path exists, for example:

```sh
ls /usr/local/llvm/lib/cmake/mlir
which mlir-tblgen
```

Project build
-------------

From the repository root:

```sh
mkdir -p build && cd build
cmake -G Ninja .. -DMLIR_DIR=/usr/local/llvm/lib/cmake/mlir -DCMAKE_BUILD_TYPE=Release
cmake --build . -j $(nproc)
```

If you installed MLIR/LLVM to a different prefix, replace `/usr/local/llvm` with that prefix.

Sanity checks
-------------

- Run `build/tools/dqc-opt/dqc-opt --help` to verify the driver prints options and registered dialects.
- Try parsing an example MLIR file: `build/tools/dqc-opt/dqc-opt test/IR/dqc_ir_test.mlir -o -`.

Troubleshooting
---------------

- Error: `mlir-tblgen: command not found` — ensure the MLIR build is installed and `mlir-tblgen` in PATH.
- Error: cannot find `MLIRConfig.cmake` — pass `-DMLIR_DIR` to CMake pointing to the `lib/cmake/mlir` directory in your MLIR install.
- Linker errors for MLIR pass creation functions — avoid calling `mlir::registerAllPasses()` in the driver if your MLIR install is minimal; instead register only the passes you need.

Optional: Development Docker image
---------------------------------

Providing a reproducible Dockerfile and devcontainer that installs MLIR is outside the scope of this file, but is a recommended next step for consistent CI and contributor environments.

End of Setup


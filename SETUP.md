# Setup & Build — DQC Compiler (short)

Prerequisites (Linux)
- Install: `build-essential cmake ninja-build git python3 clang lld`.
- Have a working LLVM/MLIR installation or build LLVM with MLIR.

Quick build steps
```bash
# from repo root
mkdir -p build && cd build
cmake -G Ninja .. -DLLVM_DIR=$LLVM_DIR -DMLIR_DIR=$MLIR_DIR -DCMAKE_BUILD_TYPE=Debug
# generate TableGen outputs (critical)
ninja DQCIncGen
# build project
ninja
# run tests
ctest --output-on-failure
```

Notes
- If you see missing `*.inc` headers, ensure `mlir-tblgen` is available and rerun `ninja DQCIncGen`.
- On macOS or Windows, adapt package manager and compilers per platform.

Troubleshooting (common)
- `mlir-tblgen: command not found` → add LLVM/MLIR bin to PATH.
- Missing `DQCOps.h.inc` → run `ninja DQCIncGen` in `build`.
# DQC Compiler: Complete Setup & Installation Guide

## Current Project Status

- **Status:** Implementation largely complete; integration & testing pending.
- **Completion:** 90% complete
- **Notes:** Use this guide to install required dependencies and build tools; after environment setup run `ninja DQCIncGen` and `ninja` to complete generated sources and build.

This guide provides step-by-step instructions to build the DQC Compiler on Linux, WSL2, and Windows.

## Table of Contents

- [Option 1: WSL2 + Ubuntu (Recommended)](#option-1-wsl2--ubuntu-recommended)
- [Option 2: Native Linux](#option-2-native-linux)
- [Option 3: macOS](#option-3-macos)
- [Option 4: Windows Native + Visual Studio](#option-4-windows-native--visual-studio)
- [Troubleshooting](#troubleshooting)

---

## Option 1: WSL2 + Ubuntu (Recommended)

### Step 1: Set Up WSL2

```bash
# On Windows (PowerShell as Administrator)
wsl --install
wsl --list --verbose
# Ensure WSL 2 is set as default
wsl --set-default-version 2
```

Open Ubuntu terminal (from Start menu or `wsl -d Ubuntu`).

### Step 2: Install Dependencies

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  python3 \
  python3-pip \
  clang \
  lld \
  pkg-config \
  libelf-dev \
  libncurses-dev \
  wget \
  curl \
  ccache
```

### Step 3: Build LLVM/MLIR from Source

```bash
# Clone LLVM project
cd ~
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# Checkout stable version
git checkout llvmorg-16.0.0

# Create build directory
mkdir build && cd build

# Configure CMake
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_INSTALL_PREFIX=/usr/local/llvm \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_CCACHE_BUILD=ON

# Build (this takes ~30-60 minutes)
ninja -j$(nproc)

# Install
sudo ninja install
```

### Step 4: Set Environment Variables

Add these to `~/.bashrc` or `~/.zshrc`:

```bash
export LLVM_DIR=/usr/local/llvm/lib/cmake/llvm
export MLIR_DIR=/usr/local/llvm/lib/cmake/mlir
export PATH=/usr/local/llvm/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/llvm/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=/usr/local/llvm:$CMAKE_PREFIX_PATH
```

Then source:
```bash
source ~/.bashrc
```

### Step 5: Verify Installation

```bash
mlir-opt --version
# Should output MLIR version 16.x.x
```

### Step 6: Build DQC Compiler

```bash
cd /workspaces/dqc-compiler
mkdir -p build && cd build

cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_DIR=$LLVM_DIR \
  -DMLIR_DIR=$MLIR_DIR

ninja -j$(nproc)
```

### Step 7: Run Tests

```bash
ctest --output-on-failure
```

---

## Option 2: Native Linux

Identical to Option 1 (WSL2). Follow all steps above, replacing `sudo apt` with your distro's package manager if needed:

### For Fedora/RHEL:
```bash
sudo dnf install -y \
  gcc gcc-c++ make cmake ninja-build \
  git python3 clang lld \
  libelf-devel ncurses-devel ccache
```

### For Arch:
```bash
sudo pacman -S \
  base-devel cmake ninja git python \
  clang lld pkg-config elfutils ccache
```

---

## Option 3: macOS

### Step 1: Install Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Dependencies

```bash
brew install cmake ninja git llvm@16 python3 ccache

# Create symlinks for LLVM tools
export LLVM_PATH=$(brew --prefix llvm@16)
export PATH=$LLVM_PATH/bin:$PATH
```

### Step 3: Build DQC Compiler

```bash
cd /path/to/dqc-compiler
mkdir -p build && cd build

export LLVM_DIR=$(brew --prefix llvm@16)/lib/cmake/llvm
export MLIR_DIR=$(brew --prefix llvm@16)/lib/cmake/mlir

cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_DIR=$LLVM_DIR \
  -DMLIR_DIR=$MLIR_DIR \
  -DCMAKE_C_COMPILER=$(brew --prefix llvm@16)/bin/clang \
  -DCMAKE_CXX_COMPILER=$(brew --prefix llvm@16)/bin/clang++

ninja -j$(nproc)
ctest --output-on-failure
```

---

## Option 4: Windows Native + Visual Studio

### Step 1: Install Visual Studio 2022

Download and install from https://visualstudio.microsoft.com/downloads/

In the installer, select:
- **Desktop development with C++**
- Include: MSVC v143, Windows SDK, CMake tools

### Step 2: Install Build Tools

Download and install:
- **CMake:** https://cmake.org/download/
- **Ninja:** Download from https://github.com/ninja-build/ninja/releases
  - Extract to `C:\Program Files\Ninja`
- **Git:** https://git-scm.com/download/win
- **Python 3:** https://www.python.org/downloads/

### Step 3: Install LLVM/MLIR

Option A: Use prebuilt binaries (easiest)
```powershell
# Download LLVM 16 installer from
# https://github.com/llvm/llvm-project/releases
# Run installer, choose installation directory (e.g., C:\LLVM16)
```

Option B: Build from source (advanced, ~1-2 hours)
```powershell
# In PowerShell as Administrator
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-16.0.0
mkdir build; cd build

cmake -G "Visual Studio 17" ..\llvm `
  -DLLVM_ENABLE_PROJECTS="mlir" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_INSTALL_PREFIX="C:\Program Files\LLVM16"

cmake --build . --config Release -j 4
cmake --install .
```

### Step 4: Set Environment Variables

Open "Edit environment variables for your account" in Windows:

Add:
- `LLVM_DIR` = `C:\LLVM16\lib\cmake\llvm` (or your LLVM install path)
- `MLIR_DIR` = `C:\LLVM16\lib\cmake\mlir`
- `Path` = Add `C:\Program Files\Ninja` and `C:\LLVM16\bin`

### Step 5: Build DQC Compiler

Open "x64 Native Tools Command Prompt for VS 2022" (from Start menu):

```powershell
cd C:\path\to\dqc-compiler
mkdir build && cd build

cmake -G Ninja .. ^
  -DCMAKE_BUILD_TYPE=Debug ^
  -DLLVM_DIR=%LLVM_DIR% ^
  -DMLIR_DIR=%MLIR_DIR%

ninja -j 4
ctest --output-on-failure
```

---

## Troubleshooting

### Error: `CMake was unable to find ... Ninja`

**Solution:** Ensure Ninja is installed and on PATH.

```bash
# Linux/WSL
which ninja
sudo apt install ninja-build

# Windows
where ninja
# If not found, download from https://github.com/ninja-build/ninja/releases
```

### Error: `mlir-tblgen: command not found`

**Solution:** MLIR tools not on PATH.

```bash
# Linux/WSL
export PATH=/usr/local/llvm/bin:$PATH

# macOS
export PATH=$(brew --prefix llvm@16)/bin:$PATH

# Windows
# Ensure C:\LLVM16\bin is in PATH environment variable
```

### Error: Missing `DQCOps.h.inc`

**Solution:** TableGen generation failed. Regenerate:

```bash
cd build
ninja DQCIncGen
```

If still missing, ensure `mlir-tblgen` is on PATH (see above).

### Error: `LLVM_DIR` or `MLIR_DIR` not found

**Solution:** Set CMake variables explicitly:

```bash
cmake .. \
  -DLLVM_DIR=/usr/local/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/usr/local/llvm/lib/cmake/mlir
```

Or check installation path:

```bash
# Find LLVM installation
find /usr -name "LLVMConfig.cmake" 2>/dev/null
```

### Error: Compilation fails with undefined references

**Solution:** Ensure LD_LIBRARY_PATH includes LLVM:

```bash
export LD_LIBRARY_PATH=/usr/local/llvm/lib:$LD_LIBRARY_PATH
# Or on macOS:
export DYLD_LIBRARY_PATH=/usr/local/llvm/lib:$DYLD_LIBRARY_PATH
```

### Slow build?

**Optimization:**

```bash
# Use ccache for faster incremental builds
ninja -j$(nproc)   # Parallel compilation

# Or use prebuilt LLVM instead of building from source
# (see Option 1, Step 3 for prebuilt binary links)
```

---

## Quick Reference: All-in-One Bash Script (WSL2/Linux)

Save as `install_dqc.sh`:

```bash
#!/bin/bash
set -e

echo "Installing DQC Compiler dependencies..."

# Install system packages
sudo apt update
sudo apt install -y build-essential cmake ninja-build git python3 python3-pip \
  clang lld pkg-config libelf-dev libncurses-dev ccache

# Build LLVM/MLIR
echo "Building LLVM/MLIR (this may take 30-60 minutes)..."
cd ~
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-16.0.0
mkdir build && cd build

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_INSTALL_PREFIX=/usr/local/llvm \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_CCACHE_BUILD=ON

ninja -j$(nproc)
sudo ninja install

# Set env vars
echo "
export LLVM_DIR=/usr/local/llvm/lib/cmake/llvm
export MLIR_DIR=/usr/local/llvm/lib/cmake/mlir
export PATH=/usr/local/llvm/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/llvm/lib:\$LD_LIBRARY_PATH
" >> ~/.bashrc

# Build DQC Compiler
echo "Building DQC Compiler..."
cd /workspaces/dqc-compiler
mkdir -p build && cd build

source ~/.bashrc

cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_DIR=$LLVM_DIR \
  -DMLIR_DIR=$MLIR_DIR

ninja -j$(nproc)

# Run tests
echo "Running tests..."
ctest --output-on-failure

echo "✅ DQC Compiler build complete!"
```

Run with:
```bash
chmod +x install_dqc.sh
./install_dqc.sh
```

---

## Verifying Installation

After building, verify everything works:

```bash
cd /workspaces/dqc-compiler/build

# Test dialect parsing
mlir-opt ../test/IR/dqc_ir_test.mlir

# Run phase tests
mlir-opt ../test/Passes/interaction_graph_test.mlir --dqc-interaction-graph

# Full test suite
ctest --output-on-failure
```

---

## Next Steps

Once installation is complete:

1. Read [README.md](./README.md) for architecture overview
2. Run the example pipeline from README's "Usage Examples" section
3. Explore individual pass implementations in `lib/Passes/`
4. Contribute! See README's "Contributing" section

---

**Last Updated:** December 28, 2025

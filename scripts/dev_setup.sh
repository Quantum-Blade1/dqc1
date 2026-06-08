#!/usr/bin/env bash
set -euo pipefail

# dev_setup.sh - Bootstrap a reproducible LLVM/MLIR toolchain for DQC1
#
# This script pins and checks out a specific `llvm-project` tag and can
# optionally build a local LLVM/MLIR toolchain with deterministic CMake flags.
# Use this on a clean Ubuntu environment to reproduce developer builds.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TOOLS_DIR=${ROOT_DIR}/tools
LLVM_PROJECT_DIR=${TOOLS_DIR}/llvm-project
INSTALL_DIR=${TOOLS_DIR}/llvm-install
PIN_FILE=${TOOLS_DIR}/llvm_version.txt

LLVM_TAG=${LLVM_TAG:-$(cat "${PIN_FILE}")}
BUILD_LLVM=0

usage() {
  cat <<EOF
Usage: $0 [--build]

Options:
  --build    Build and install LLVM/MLIR into ${INSTALL_DIR} (time-consuming).
  --tag TAG  Override pinned tag from ${PIN_FILE}.
  --help     Show this message.

This script clones llvm-project at a pinned tag and prints environment
exports to add the toolchain to PATH. To build LLVM/MLIR, pass --build.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) BUILD_LLVM=1; shift ;;
    --tag) LLVM_TAG="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

echo "Pinned llvm-project tag: ${LLVM_TAG}"

mkdir -p "${TOOLS_DIR}"

if [ ! -d "${LLVM_PROJECT_DIR}" ]; then
  echo "Cloning llvm-project (${LLVM_TAG}) into ${LLVM_PROJECT_DIR}"
  git clone --depth 1 --branch "${LLVM_TAG}" https://github.com/llvm/llvm-project.git "${LLVM_PROJECT_DIR}"
else
  echo "llvm-project already exists at ${LLVM_PROJECT_DIR}; fetching and checking out ${LLVM_TAG}"
  pushd "${LLVM_PROJECT_DIR}" >/dev/null
  git fetch --depth 1 origin "${LLVM_TAG}" || true
  git checkout --force "${LLVM_TAG}" || git checkout "${LLVM_TAG}"
  popd >/dev/null
fi

echo "Writing pinned tag to ${PIN_FILE}"
echo "${LLVM_TAG}" > "${PIN_FILE}"

echo "To use prebuilt clang/llvm if available, set PATH to include ${INSTALL_DIR}/bin"

if [ "${BUILD_LLVM}" -eq 1 ]; then
  echo "Starting deterministic build of LLVM/MLIR (this may take a long time)"
  mkdir -p "${INSTALL_DIR}"
  BUILD_DIR="${TOOLS_DIR}/llvm-project/build"
  mkdir -p "${BUILD_DIR}"
  pushd "${BUILD_DIR}" >/dev/null

  # Reproducible compiler flags: strip absolute paths from debug info and
  # map file paths to a stable prefix. These flags help reduce environment
  # dependent artifacts but do not guarantee bit-for-bit reproducibility
  # across toolchains.
  FF_PREFIX_MAP="-ffile-prefix-map=${ROOT_DIR}=." 
  CMAKE_FLAGS=(
    -G Ninja
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DLLVM_ENABLE_PROJECTS="clang;mlir"
    -DLLVM_ENABLE_RUNTIMES=""
    -DLLVM_ENABLE_ASSERTIONS=ON
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
    -DCMAKE_C_FLAGS="${FF_PREFIX_MAP}"
    -DCMAKE_CXX_FLAGS="${FF_PREFIX_MAP}"
    -DLLVM_USE_LINKER=lld
  )

  echo "Configuring LLVM with: ${CMAKE_FLAGS[*]}"
  cmake "${LLVM_PROJECT_DIR}/llvm" "${CMAKE_FLAGS[@]}"

  echo "Building LLVM/MLIR"
  ninja -j"$(nproc)"

  echo "Installing into ${INSTALL_DIR}"
  ninja install
  popd >/dev/null

  echo "LLVM/MLIR build and install complete. Add the following to your shell:"
  echo "  export PATH=\"${INSTALL_DIR}/bin:\$PATH\""
fi

cat <<EOF
Quick usage notes:

- To use the pinned llvm/clang/MLIR build (if you built it):
    export PATH="${INSTALL_DIR}/bin:":\$PATH

- Recommended CMake flags for reproducible builds of DQC1:
    cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_C_FLAGS="-ffile-prefix-map=${ROOT_DIR}=." \
      -DCMAKE_CXX_FLAGS="-ffile-prefix-map=${ROOT_DIR}=." \
      -DDQC_ENABLE_TESTS=ON ..

EOF

echo "dev_setup.sh finished"

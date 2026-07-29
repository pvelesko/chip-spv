#!/bin/bash

# Test description:
# Same as TestStaticLibRDC.bash, but with an explicit --offload-arch=gfx1030 on
# every hipcc invocation, as AMD-derived build systems (Kokkos, hipBLAS, ...)
# do unconditionally.
#
# The offload arch renames the offload bundle entry from
# "hip-spirv64-unknown-unknown-unknown-generic" to "...-gfx1030". clang's
# HIPSPV toolchain unbundles *static archives* with a hard-coded "generic"
# target ID, so with a non-generic arch the archive's device bundles no longer
# match, clang-offload-bundler (run with -allow-missing-bundles) produces an
# empty device archive, and every kernel that reaches the link only as a member
# of a .a silently disappears from the device module.
# See https://github.com/CHIP-SPV/chipStar/issues/1408.

# Exit script on error
set -eu

# CMake substituted variables
SRC_DIR="@CMAKE_CURRENT_SOURCE_DIR@"
OUT_DIR="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"
HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"

ARCH_FLAG="--offload-arch=gfx1030"

# Create output directory
mkdir -p "${OUT_DIR}"

# Compile the device code files
${HIPCC} ${ARCH_FLAG} -fgpu-rdc -fPIC -I"${SRC_DIR}" -c "${SRC_DIR}/k.cu" -o "${OUT_DIR}/k.o"
${HIPCC} ${ARCH_FLAG} -fgpu-rdc -fPIC -I"${SRC_DIR}" -c "${SRC_DIR}/k1.cu" -o "${OUT_DIR}/k1.o"

# Create the static library
ar rcs "${OUT_DIR}/libk.a" "${OUT_DIR}/k.o" "${OUT_DIR}/k1.o"

# Compile the main host file
${HIPCC} ${ARCH_FLAG} -fgpu-rdc -I"${SRC_DIR}" -c "${SRC_DIR}/t.cpp" -o "${OUT_DIR}/t.o"

# Link the main file and the static library
${HIPCC} ${ARCH_FLAG} -fgpu-rdc "${OUT_DIR}/t.o" "${OUT_DIR}/libk.a" \
         -o "${OUT_DIR}/TestStaticLibRDCOffloadArch"

RUN_EXEC="${OUT_DIR}/TestStaticLibRDCOffloadArch"
echo "Running: ${RUN_EXEC}"
STDERR_OUTPUT=$("${RUN_EXEC}" 2>&1) || true

# The kernels live only in libk.a. If they were dropped from the device module
# the launch fails with "Failed to find kernel via kernel name".
if echo "${STDERR_OUTPUT}" | grep -qE 'CHIP error|hipError'; then
  echo "Test FAILED: Error messages found in output."
  echo "Output:"
  echo "${STDERR_OUTPUT}"
  exit 1
fi

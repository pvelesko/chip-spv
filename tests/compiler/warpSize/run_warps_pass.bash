#!/bin/bash
# Check that HipWarpsPass pins intel_reqd_sub_group_size on every kernel.
#
# Usage: run_warps_pass.bash <input.ll>
#
# The input is a device module that calls no warp primitive at all. It is still
# warp-width sensitive: it is a barrier-free shared-memory reduction that relies
# on warp lock-step semantics. HipWarpsPass used to return early unless a warp
# primitive declaration happened to be present in the module, which left such
# kernels unpinned and let the compiler pick a subgroup narrower than warpSize.
# See CHIP-SPV/chipStar#1409.

set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 <input.ll>"
  exit 1
fi

INPUT_FILE="$1"
BASE_NAME=$(basename "${INPUT_FILE}" .ll)
OUTPUT_BC="${BASE_NAME}.warps.bc"
OUTPUT_LL="${BASE_NAME}.warps.ll"

"${LLVM_OPT}" -load-pass-plugin "${HIP_SPV_PASSES_LIB}" \
  -passes=hip-post-link-passes "${INPUT_FILE}" -o "${OUTPUT_BC}"
"${LLVM_DIS}" "${OUTPUT_BC}" -o "${OUTPUT_LL}"

KERNELS=$(grep -c 'spir_kernel' "${OUTPUT_LL}" || true)
PINNED=$(grep -c 'intel_reqd_sub_group_size' "${OUTPUT_LL}" || true)

echo "kernels=${KERNELS} pinned=${PINNED}"

if [ "${KERNELS}" -eq 0 ]; then
  echo "ERROR: no spir_kernel survived the pass pipeline; test input is stale"
  exit 1
fi

if [ "${PINNED}" -ne "${KERNELS}" ]; then
  echo "ERROR: ${KERNELS} kernel(s) but only ${PINNED} carry"
  echo "       intel_reqd_sub_group_size. A kernel whose subgroup size is not"
  echo "       pinned may be compiled narrower than warpSize, which silently"
  echo "       breaks warp lock-step device code."
  echo "See ${OUTPUT_LL} for details"
  exit 1
fi

exit 0

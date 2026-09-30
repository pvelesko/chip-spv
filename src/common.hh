/*
 * Copyright (c) 2021-22 chipStar developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef HIP_COMMON_HH
#define HIP_COMMON_HH

#include "chipStarConfig.hh"
#include "SPIRVFuncInfo.hh"

#include <map>
#include <set>
#include <vector>
#include <stdint.h>
#include <string>
#include <memory>
#include <unordered_set>
#include <utility>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <queue>
#include <stack>

using SPVFunctionInfoMap = std::map<std::string, std::shared_ptr<SPVFuncInfo>>;

/// Phase H4 (Vulkan path): a module-scope `__device__` global lowered by
/// the HIPSPVLowerToHLSLShape pass into a *per-kernel hidden kernel-arg*
/// (set=0). Recovered by walking OpName decorations on set=0 StorageBuffer
/// descriptors whose name matches the `__hipspv_dg_<symbol>` prefix. Used
/// by the chipStar runtime to populate its host-symbol -> device-buffer map
/// for hipMemcpyToSymbol and to bind the corresponding cl_mem at kernel
/// launch via `clSetKernelArg`.
struct SPVDeviceGlobal {
  std::string Name;       ///< Original `__device__` variable name (e.g. "A").
  uint32_t Set = 0;       ///< Descriptor set (always 0 in the H4 path).
  uint32_t Binding = 0;   ///< Descriptor binding within set 0 (legacy; per-kernel binding lives in HiddenArgsByKernel below).
  size_t Size = 0;        ///< Size in bytes of the underlying element type.
  /// Phase I3: initial-value bytes recovered from the bridging pass via
  /// the `__hipspv_dg_<sym>__sz_<size>__init_<hex>` OpName encoding. The
  /// runtime copies these into the per-symbol device buffer at allocate
  /// time and on every hipDeviceReset() to satisfy the HIP semantics that
  /// `__device__ int A = 123;` reads back as 123 unless explicitly written.
  std::vector<uint8_t> InitData;
};

/// Phase H4: per-kernel record of which device-global symbol is bound at
/// which kernel-argument ordinal. Populated from the SPV's `__hipspv_dg_*`
/// OpName decorations during analyzeSPIRV.
struct SPVKernelDeviceGlobalArg {
  std::string Symbol;     ///< `__device__` variable name.
  uint32_t ArgIndex = 0;  ///< Kernel-arg ordinal (post-bridging-pass).
};

struct SPVModuleInfo {
  SPVFunctionInfoMap FuncInfoMap;

  /// Set to true if the module is known not to have indirect global
  /// buffer accesses (IGBA) in any kernel.
  bool HasNoIGBAs = false;

  /// Phase H4: device globals lowered to per-kernel hidden args. Empty in
  /// OCL / non-Vulkan modules.
  std::vector<SPVDeviceGlobal> DeviceGlobals;

  /// Phase H4: kernel-name -> ordered list of (symbol, kernel-arg-index)
  /// hidden device-global args. The runtime uses this map at launch to
  /// bind the chipStar-allocated cl_mem of `<symbol>` at the recorded ord.
  std::map<std::string, std::vector<SPVKernelDeviceGlobalArg>>
      HiddenDGArgsByKernel;

  /// Phase Z3 (Vulkan BDA): kernel-name -> ordered list of byte offsets
  /// within the kernel's push-constant block where an 8-byte Buffer Device
  /// Address slot lives. The bridging pass (rewriteKernelSignatureBDA) emits
  /// these via NonSemantic.ClspvReflection ArgumentPointerPushConstant=26
  /// ExtInsts; inject_reflection.py produces them when it sees a global
  /// `.hipspv.bda_offsets.<kernel>` constant. The runtime substitutes the
  /// raw HIP pointer at each offset with vkGetBufferDeviceAddress(buffer)
  /// before pushing the PC block. Empty for non-BDA modules.
  std::map<std::string, std::vector<uint32_t>> BDAPointerSlotOffsetsByKernel;
  /// Per kernel: (HIP argument index, push-constant offset) of each i32 the
  /// runtime sets to (pointer argument != nullptr).
  std::map<std::string, std::vector<std::pair<uint32_t, uint32_t>>>
      NullFlagSlotsByKernel;
};

// Processing done before analysis.
bool preprocessSPIRV(const char *Bytes, size_t NumBytes,
                     bool PreventNameDemangling, std::vector<uint32_t> &Dst);
bool analyzeSPIRV(uint32_t *Stream, size_t NumWords, SPVModuleInfo &ModuleInfo);
// Processing done after analysis.
bool postprocessSPIRV(std::vector<uint32_t> &Binary);

/// A prefix given to lowered global scope device variables. No shadow kernel
/// name may start with it.
constexpr char ChipVarPrefix[] = "__chip_var_addr_";
/// A prefix used for a shadow kernel used for querying device
/// variable properties.
constexpr char ChipVarInfoPrefix[] = "__chip_var_info_";
/// A prefix used for a shadow kernel used for binding storage to
/// device variables.
constexpr char ChipVarBindPrefix[] = "__chip_var_bind_";
/// A prefix used for a shadow kernel used for initializing device
/// variables.
constexpr char ChipVarInitPrefix[] = "__chip_var_init_";
/// The name of a single combined shadow kernel that initializes ALL
/// host-accessible program-scope variables in one launch. Emitted (in the
/// program-scope-globals lowering) instead of per-variable init kernels to
/// avoid O(N) single-work-item kernel launches. See issue #582.
constexpr char ChipVarInitAllName[] = "__chip_var_init_all";
/// A structure to where properties of a device variable are written.
/// CHIPVarInfo[0]: Size in bytes.
/// CHIPVarInfo[1]: Requested alignment.
/// CHIPVarInfo[2]: Zero if variable has no initializer. Otherwise
/// ChipVarInitGridStride if its init kernel runs on any 1-D launch geometry,
/// ChipVarInitHostFill if the runtime zeroes it instead of the init kernel, or
/// another non-zero value if it must run on a single work item.
using CHIPVarInfo = int64_t[3];
constexpr int64_t ChipVarInitGridStride = 2;
constexpr int64_t ChipVarInitHostFill = 3;
/// Zero initializers at least this large are filled by the runtime.
constexpr uint64_t ChipVarFillThreshold = 64 * 1024 * 1024;

/// The name of the shadow kernel responsible for resetting host-inaccessible
/// global device variables (e.g. static local variables in device code).
constexpr char ChipNonSymbolResetKernelName[] = "__chip_reset_non_symbols";

/// The prefix for global-scope variables in SPIR-V modules for carrying
/// information about "spilled" arguments
///
/// see HipKernelArgSpiller.cpp for details. Full name of such
/// variables is '<ChipSpilledArgsVarPrefix><kernel-name>'
constexpr char ChipSpilledArgsVarPrefix[] = "__chip_spilled_args_";

/// The prefix for global-scope annotation variables recording which device
/// globals feed a kernel's implicit trailing DeviceGlobal arguments
/// (globals-as-kernel-args lowering, used when program-scope globals are
/// disabled). The variable '<ChipGVarArgPrefix><kernel-name>' holds the
/// NUL-separated original global names in trailing-argument order. See
/// HipGlobalVariables.cpp for details.
constexpr char ChipGVarArgPrefix[] = "__chip_gvararg_";

/// The name of a global variable which indicates, when non-zero, if
/// the abort() function was called by a kernel.
constexpr char ChipDeviceAbortFlagName[] = "__chipspv_abort_called";

/// The name of a global variable which holds the message of a failed
/// device-side assertion (__chipspv_abort_msg in include/hip/spirv_hip.hh),
/// and the byte offset of its NUL-terminated text: the bytes before the text
/// hold the claim word with which the device picks a single writer.
constexpr char ChipDeviceAbortMsgName[] = "__chipspv_abort_msg";
constexpr size_t ChipDeviceAbortMsgTextOffset = sizeof(int);

/// The name of a global variable which is the device heap.
constexpr char ChipDeviceHeapName[] = "__chipspv_device_heap";

#endif

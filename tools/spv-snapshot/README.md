# SPV Snapshot — Reference for Phase 2/3 Agents

Snapshot of the SPIR-V blob produced by the HIP -> Vulkan toolchain for a
plain `VecAdd` kernel. This is the exact format the new native Vulkan backend
(`CHIPBackendVulkan`) must consume directly — no clvk in the pipeline.

## Provenance

| Field | Value |
|---|---|
| Date captured | 2026-05-11 |
| Sample | `samples/2_vecadd` (simplified — see notes below) |
| Source | `vecadd-source.cpp` in this directory |
| Toolchain | `/space/pvelesko/install/llvm/23.0-native-vulkan/bin/clang++` (LLVM 23.0.0 git f8318fa6, branch `vulkan-spirv-23`) |
| chipStar install | `/space/pvelesko/install/HIP/chipStar/vulkanize-native/` |
| Compile cmd | `hipcc -c VecAddSimple.cpp --offload-arch=spirv64-unknown-chipstar-chipstarvulkan` |
| Extracted with | `spirv-extractor` from the chipStar build (reads the embedded fatbin in the `.o`) |
| Generator | `LLVM LLVM SPIR-V Backend; 23` |
| SPIR-V version | 1.4 |
| Bound | 194 |

### Why a simplified vecadd

The shipped `samples/2_vecadd/VecAdd.cpp` uses `HIP_vector_type<float, 4>` and
triggers a known LLVM 23 backend assertion in
`SPIRVLegalizePointerCast::buildLegalizedStore` ("Failed to store to aggregate:
Could not find compatible memory layout") on this toolchain build. To get a
clean, end-to-end reference SPV for the spike, this snapshot was captured from
a plain `float*` kernel (semantically identical signature pattern to what the
spike must run). Phase 4 integration on the original `2_vecadd` will require
upstream fix of the legalize-pointer-cast crash OR a workaround in the kernel
source; documented separately.

## What Phase 2 / Phase 3 agents should look at

### Capabilities (line 6-7)

```
OpCapability Shader
OpCapability Int8
```

That is the entire capability list for a plain float-only kernel. The wider
extension set the plan enumerates (`VK_KHR_buffer_device_address`,
`VK_EXT_shader_atomic_float`, `VK_KHR_shader_clock`, `VK_KHR_16bit_storage`,
`VK_KHR_shader_atomic_int64`) is conditional on kernel content — they will
appear only when the source uses BDA pointers / atomics / 16-bit types /
shader-clock intrinsics. For the 2_vecadd spike none of those are needed.

`Int8` is unconditional here because the bridging pass emits per-arg name
strings as `uchar` arrays inside the entry function (see `OpName
%_hipspv_argname_*` lines 17-19 and `OpConstantComposite ... uchar_*`
sequences at lines 90-92). I1 must therefore unconditionally request
`VK_KHR_storage_buffer_storage_class` (1.1 core) and the device feature
`shaderInt8` even on the simplest kernel. `VK_KHR_8bit_storage` is **not**
required because the int8 array lives in `Function` storage, not Storage
Buffer / Uniform — verify this assumption for kernels with byte-typed
descriptor arguments.

### Extensions (line 8)

```
OpExtension "SPV_KHR_non_semantic_info"
```

Only one. This is for the ClspvReflection ext-inst set below. I1 must enable
the corresponding Vulkan extension **`VK_KHR_shader_non_semantic_info`**
(promoted to Vulkan 1.3 core, but the spike targets 1.2 + extension).

### Extended instruction set (line 9)

```
%176 = OpExtInstImport "NonSemantic.ClspvReflection.5"
```

The reflection set ID (`%176` here, but renumbers per module). F2's lifted
parser must locate this ExtInstImport, capture its result-id, then scan for
every `OpExtInst %void %<that-id> <opcode> ...` instruction in the module
(typically clustered at the tail — lines 148-152 here).

### Memory model (line 10)

```
OpMemoryModel Logical GLSL450
```

Plain Logical addressing. **No PhysicalStorageBuffer64** here — that would
appear only if the bridging pass emitted BDA pointers. I3 must build pipeline
layouts assuming descriptor-set bindings as the primary memory access path.

### Entry point (line 11)

```
OpEntryPoint GLCompute %_Z12VecAddKernelPKfS0_Pfi "_Z12VecAddKernelPKfS0_Pfi"
  <list of interface variables>
```

The entry point name is the Itanium-mangled kernel name (NOT the demangled
`VecAddKernel`). H1's `Kernel::compile` MUST use `vkGetPipelineCacheData` /
`vkCreateComputePipelines` with `pName = "_Z12VecAddKernelPKfS0_Pfi"` — same
string the reflection `Kernel` instruction declares (see %178 on line 14 and
the reflection cluster below).

### Execution mode (line 12)

```
OpExecutionModeId %<entry> LocalSizeId %190 %191 %192
```

`LocalSizeId` (not `LocalSize`) — workgroup size is **spec-constant**, not
baked into the SPV. The three spec-constants are decorated `SpecId 0/1/2`
(lines 33-35) and declared as `OpSpecConstant %uint 1` (lines 103-105, default
of 1,1,1). I3/I4 must specialize these via `VkSpecializationInfo` at
pipeline-creation time with the actual `hipModuleLaunchKernel` block-size
(`x, y, z` -> `SpecId 0, 1, 2`).

### Descriptor layout — storage-buffer kernel args (lines 40-45)

```
OpDecorate %_Z12VecAddKernelPKfS0_Pfi_0 DescriptorSet 0  Binding 0
OpDecorate %_Z12VecAddKernelPKfS0_Pfi_1 DescriptorSet 0  Binding 1
OpDecorate %_Z12VecAddKernelPKfS0_Pfi_2 DescriptorSet 0  Binding 2
```

All buffer arguments land in `DescriptorSet 0` with bindings indexed by
positional kernel-arg order (skipping POD args). Each is typed
`OpTypePointer StorageBuffer %_struct_10` where `_struct_10` is `{
runtimearr<float> }` — i.e., `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` with
`VK_DESCRIPTOR_BINDING_FLAGS_NONE`. I3 builds the
`VkDescriptorSetLayoutBinding` array from these decorations OR (preferred)
from the ClspvReflection `ArgumentStorageBuffer` entries below.

### Push-constant POD args (line 64, 89)

```
%_ptr_PushConstant__Z12VecAddKernelPKfS0_Pfi_pcs = OpTypePointer PushConstant ...
%_hipspv_pcs__Z12VecAddKernelPKfS0_Pfi_1 = OpVariable ... PushConstant
```

The single POD `int n` argument is packed into a push-constant block
`_Z12VecAddKernelPKfS0_Pfi_pcs` containing one `uint` at offset 0. Total push
constant size for this kernel: **4 bytes**. I4's `ExecItem::setupAllArgs`
gathers all POD args into the push-constant block laid out per the reflection
`ArgumentPodPushConstant` entry below.

### Reflection metadata cluster (lines 148-152) — the gold

```
%179 = OpExtInst %void %176 Kernel                     %<entry-fn> %178("kernel_name") %uint_4 %uint_0 %177("")
%182 = OpExtInst %void %176 ArgumentStorageBuffer      %179 %uint_0 %uint_0 %uint_0   ; arg-index=0, ord=0, descriptor-set=0, binding=0
%184 = OpExtInst %void %176 ArgumentStorageBuffer      %179 %uint_1 %uint_0 %uint_1   ; arg-index=1, ord=0, descriptor-set=0, binding=1
%186 = OpExtInst %void %176 ArgumentStorageBuffer      %179 %uint_2 %uint_0 %uint_2   ; arg-index=2, ord=0, descriptor-set=0, binding=2
%188 = OpExtInst %void %176 ArgumentPodPushConstant    %179 %uint_3 %uint_0 %uint_4   ; arg-index=3, push-constant-offset=0, size=4
```

ClspvReflection.5 schema (relevant subset observed):

| ExtInst opcode | Operands (after result-type and reflection-set ID) |
|---|---|
| `Kernel` | `<entry-fn-id> <name-string-id> <#args> <flags> <attrs-string-id>` |
| `ArgumentStorageBuffer` | `<kernel-handle> <arg-ordinal> <descriptor-set> <binding>` |
| `ArgumentPodPushConstant` | `<kernel-handle> <arg-ordinal> <pc-offset> <pc-size>` |

F2's lifted parser (`tryAnalyzeVulkanReflection`) emits these into
`SPVKernelDeviceGlobalArg` / `SPVModuleInfo`. I3 consumes those to build
`VkDescriptorSetLayout` + `VkPipelineLayout` (`pushConstantRanges` populated
from the `ArgumentPodPushConstant` offsets/sizes).

## Surprises vs the plan's "Required Vulkan Extensions" list

The plan enumerates 8 Vulkan extensions as baseline:

| Plan list | Appears here? | Notes |
|---|---|---|
| `VK_KHR_buffer_device_address` | No (Logical addressing only) | Bridging pass uses descriptor-set + push-constant model, not BDA. Could surface for `hipDeviceptr_t` round-trips; defer |
| `VK_EXT_shader_atomic_float` | No | Kernel has no atomics |
| `VK_KHR_shader_clock` | No | No `clock()` builtin |
| `VK_KHR_16bit_storage` | No | No `__half` |
| `VK_KHR_shader_non_semantic_info` | **Yes** (only one observed) | Required for the ClspvReflection set |
| `VK_KHR_storage_buffer_storage_class` | Implicit (Vulkan 1.1 core) | StorageBuffer storage class used everywhere |
| `VK_KHR_shader_float_controls` | No | No special FP-mode decorations |
| `VK_KHR_shader_atomic_int64` | No | No 64-bit atomics |

**Net:** for the spike target (`2_vecadd`), I1 must enable exactly
`VK_KHR_shader_non_semantic_info` plus the device feature `shaderInt8` (for
the per-arg name strings the bridging pass emits). The wider extension set is
content-driven and should be enabled lazily (queried per-module in
`Module::compile`) rather than unconditionally at device creation, OR
unconditionally requested at device creation with a feature-gate on
`VkPhysicalDevice*Features2` chain (closer to clvk's behavior).

`Shader` capability is universal for all compute pipelines (Vulkan 1.0 core).

## Volk recommendation

**Skip volk for the spike.** F1's `find_package(Vulkan 1.2 REQUIRED)` with
direct linkage against `${Vulkan_LIBRARY}` is the simplest path and matches
how every chipStar dependency (OpenCL, Level0) is wired today. Volk's
runtime-dispatch benefits (loader bypass, per-device function tables) become
relevant when:

1. chipStar wants to gracefully degrade when `libvulkan.so.1` is absent at
   runtime (today: hard link, hard fail). Mirrors the OpenCL backend which
   *also* hard-links `libOpenCL.so`.
2. chipStar wants per-physical-device function pointer caching to shave
   dispatch overhead. Negligible at the kernel-launch granularity HIP
   programs operate at.

Neither matters for Phase 4. Re-evaluate in Phase 5 polish.

## Reproducing this snapshot

```bash
export HIP_PATH=/space/pvelesko/install/HIP/chipStar/vulkanize-native
export PATH=/space/pvelesko/install/llvm/23.0-native-vulkan/bin:$PATH

# Compile the simplified vecadd
$HIP_PATH/bin/hipcc -c vecadd-source.cpp -o /tmp/VecAddSimple.o \
  --offload-arch=spirv64-unknown-chipstar-chipstarvulkan

# Extract embedded SPIR-V (prints disassembly to stdout)
$HIP_PATH/bin/spirv-extractor /tmp/VecAddSimple.o > vecadd.spvasm
```

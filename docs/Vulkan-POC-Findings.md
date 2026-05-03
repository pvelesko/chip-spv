# HIP → Vulkan SPIR-V via clang: POC Findings

## Goal

Determine whether modern clang/LLVM (the in-tree SPIR-V backend on LLVM 22)
can directly emit Vulkan-flavored SPIR-V from HIP source, removing the need
for the post-process translator at `vulkanize/src/spirv_to_vulkan.cpp`
(4,095 lines).

## Build

- Worktree: `/space/pvelesko/llvm-project/vulkan-spirv` (branch
  `vulkan-spirv` off `chipStar-llvm-22` @ `f92ff9fac3b7`).
- Build: `/space/pvelesko/llvm-project/vulkan-spirv/build`
  (Release+asserts, X86 + experimental SPIRV target, clang+lld projects).
- Bootstrap compiler: `/space/pvelesko/install/llvm/22.0-native/bin/clang`.
- `clang --version`: `clang version 22.1.0` (`f92ff9fac3b7`).
- `llc --version` registered targets: `spirv`, `spirv32`, `spirv64`.

## Probe kernel

`/tmp/poc_kernel.hip`:

```cpp
#include <hip/hip_runtime.h>
__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

HIP headers from `/space/pvelesko/install/chipstar-stream-wait-fix/include`
(the `vulkanize/HIP` submodule was not initialised).

## Experiments

### A — HIP frontend → LLVM IR (sanity)

```
clang -x hip --offload=spirv64 --offload-device-only \
      -nogpulib -nogpuinc -isystem $HIPINC -D__HIP_PLATFORM_SPIRV__ \
      -O2 -emit-llvm -S poc_kernel.hip -o /tmp/poc_dev.ll
```

**Exit 0.** IR has `target triple = "spirv64"`, kernel decl
`define hidden spir_kernel void @_Z4vaddPKfS0_Pfi(ptr addrspace(1) ...)`.
HIP frontend emits OpenCL-flavored IR: `spir_kernel` calling conv,
`addrspace(1)` for global-memory pointers.

### A2 — `llc -mtriple=spirv64-unknown-unknown` (OCL baseline backend)

```
llc -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_relaxed_printf_string_address_space \
    /tmp/poc_dev.ll -o /tmp/poc_ocl.spv -filetype=obj
```

**Exit 134.** `LLVM ERROR: Unknown function in:%48:iid = OpFunctionCall
%8:type, @__chipspv_abort, ...` — the IR references chipStar runtime
helpers (`__chipspv_abort`, `printf`) that are normally satisfied by linking
the chipStar device-bitcode library before codegen. **Not a Vulkan
problem**, just a reminder that the standalone IR isn't link-complete.
Skipped further A2 work since it isn't on the Vulkan critical path.

### B — Vulkan triple at the backend on HIP IR

```
llc -mtriple=spirv64-unknown-vulkan1.3-compute /tmp/poc_dev.ll \
    -o /tmp/poc_vk.spv -filetype=obj
```

**Exit 134.**

```
LLVM ERROR: This entry point lacks mandatory hlsl.shader attribute.
... Running pass 'IRTranslator' on function '@_Z4vaddPKfS0_Pfi'
... SPIRVCallLowering::lowerFormalArguments
```

The SPIR-V backend's Vulkan/Shader path requires every entry point to carry
a `hlsl.shader` function attribute (mapped to GLCompute / Fragment / Vertex
in `SPIRVCallLowering.cpp`). HIP's `spir_kernel` calling convention alone
does **not** satisfy this gate.

### C — Vulkan triple at clang frontend

`--offload=spirv64-unknown-vulkan1.3-compute`: **`error: unknown target
triple 'spirv64-unknown-vulkan1.3-compute'`** from the driver. The HIP
toolchain accepts only `spirv64` (no Vulkan environment).

`--offload=spirv-unknown-vulkan1.3-compute` (32-bit logical): driver
accepts the triple, but chipStar's `hip/devicelib/macros.hh:43` redefines
`size_t` as `unsigned long` while the 32-bit Vulkan triple already
typedef'd it as `unsigned int`:

```
typedef redefinition with different types ('unsigned long' vs 'unsigned int')
```

Triple acceptance probe (`clang --target=$T -x c -c -o /dev/null /dev/null`):

| Triple | Accepted? |
|---|---|
| `spirv-unknown-vulkan1.3-compute` | yes |
| `spirv32-unknown-vulkan1.3-compute` | no |
| `spirv64-unknown-vulkan1.3-compute` | no |
| `spirv64-unknown-vulkan-compute` | no |
| `spirv64v1.3-unknown-vulkan-compute` | no |

**Only the bare-`spirv-` (Logical, no width) Vulkan triple is recognised.**

For plain C++ (not HIP) on the accepted triple:

```
clang --target=spirv-unknown-vulkan1.3-compute -O2 -S poc_cxx.cpp ...
```

crashes in `SPIRVLegalizePointerCast.cpp:422` with `Unsupported ptrcast
user. Please fix.` — confirming the Vulkan path expects HLSL-shaped IR
(handle types, no general pointer arithmetic), not arbitrary C++.

### D — Hand-written minimal IR with `hlsl.shader`

```ll
target triple = "spirv64-unknown-vulkan1.3-compute"
define void @vadd(ptr addrspace(1) %a, ptr addrspace(1) %b,
                  ptr addrspace(1) %c, i32 %n) #0 { ret void }
attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="64,1,1" }
```

```
llc -mtriple=spirv64-unknown-vulkan1.3-compute poc_minimal.ll ...
```

**Exit 0.** 352-byte module. `spirv-dis`:

```
; SPIR-V Version: 1.4 ; Generator: LLVM SPIR-V Backend
OpCapability Shader / Int8
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %vadd "vadd"
OpExecutionMode %vadd LocalSize 64 1 1
%_ptr_CrossWorkgroup_uchar = OpTypePointer CrossWorkgroup %uchar
%6 = OpTypeFunction %void %_ptr_CrossWorkgroup_uchar (×3) %uint
```

Caveat: pointer types came out as `OpTypePointer CrossWorkgroup` — i.e.
the OpenCL global address space — because the backend mapped LLVM
`addrspace(1)` straight through. **Vulkan requires `StorageBuffer`,
`Workgroup`, or `Function` storage classes**, never CrossWorkgroup. This
output would not pass `spirv-val --target-env vulkan1.3`. (Note also that
`llc` accepted the `spirv64-` Vulkan triple while `clang` did not; the
backend is more permissive than the driver.)

So even when D "works", the lowering is Frankensteined: shader-flavored
top-level (memory model, execution model, entry point) wrapped around
kernel-flavored guts (CrossWorkgroup pointers, no descriptor decorations,
no push-constant lowering for scalar args).

## Defect list (driver → frontend → IR → backend)

| # | Layer | Defect | File / line | Effort |
|---|---|---|---|---|
| 1 | Driver | HIPSPV toolchain hardcodes `spirv64` and refuses Vulkan triple variants. | `clang/lib/Driver/ToolChains/HIPSPV.cpp` (≈194) | low |
| 2 | Driver | `spirv64`/`spirv32` + `vulkan*` env not registered as a valid Triple combo. | `llvm/lib/TargetParser/Triple.cpp` triple-canonicalisation | low/med |
| 3 | Frontend | HIP `__global__` → `spir_kernel` calling conv, no `hlsl.shader("compute")` attribute attached. Backend rejects with "lacks mandatory hlsl.shader attribute". | `clang/lib/Sema/SemaCUDA.cpp` (kernel-attr attach), `clang/lib/CodeGen/Targets/SPIR.cpp` (Vulkan vs OCL split), `clang/lib/CodeGen/CodeGenFunction.cpp` (attr emit) | **high** |
| 4 | Frontend | `numthreads` (LocalSize) is unknown at compile time for HIP — block dims are runtime values. Vulkan SPIR-V either bakes them into `OpExecutionMode LocalSize` or uses `WorkgroupSize` builtin via spec constants. | new lowering path; `clang/lib/CodeGen` and `SPIRVCallLowering.cpp` | **high** |
| 5 | Frontend | Pointer-arg lowering: HIP passes raw global pointers (addrspace 1). Vulkan compute wraps them in `StorageBuffer` blocks with descriptor set/binding decorations. Requires synthesising wrapper structs + Binding decorations. | `clang/lib/CodeGen/Targets/SPIR.cpp`, new pass before `SPIRVCallLowering` | **very high** |
| 6 | Frontend | Scalar args (`int n`) need to land in Vulkan push-constant block, not plain function params. | same as #5 | high |
| 7 | IR/Backend | LLVM `addrspace(1)` is mapped to SPIR-V `CrossWorkgroup` even under a Vulkan triple (Experiment D). Vulkan must use `StorageBuffer` / `Workgroup` / `Function`. | `llvm/lib/Target/SPIRV/SPIRVUtils.cpp` storage-class mapping, gated by `Subtarget.isShader()` | med |
| 8 | Frontend | chipStar headers' `size_t` typedef collides with 32-bit `spirv-` triple. Will need a 64-bit Vulkan-environment triple to even parse HIP headers. Pairs with defect #2. | `vulkanize/HIP/include/hip/devicelib/macros.hh:43` (or driver-side default-defines) | low |
| 9 | Runtime | `__chipspv_abort` / `printf` linkage gaps in standalone IR. Existing pipeline solves this via device-bitcode link before SPIR-V codegen. Not Vulkan-specific. | chipStar build system | n/a |
| 10 | Frontend | `threadIdx.{x,y,z}` / `blockIdx.{x,y,z}` / `blockDim.*` need lowering to SPIR-V builtins `LocalInvocationId`, `WorkgroupId`, `WorkgroupSize`. HIP currently lowers them via OpenCL `get_*` runtime helpers in chipStar's device lib. | new lowering pass or chipStar device-lib variant | med |

## Recommendation: **NO-GO on the "just use clang" path, at least without serious frontend work.**

The backend is closer than expected — the SPIR-V backend already has the
machinery to emit Vulkan-shaped modules (Experiment D proves this in 25
lines of IR). The blocker is **frontend** work: defects #3, #4, #5, #6 all
require teaching clang's HIP/CUDA frontend to produce HLSL-flavored entry
points (with `hlsl.shader`/`hlsl.numthreads` attributes, descriptor-set
struct wrappers for pointer args, push-constant blocks for scalars). That
is conceptually a small new toolchain mode in `HIPSPV` plus parallel
changes in `clang/lib/CodeGen/Targets/SPIR.cpp`, but the pointer-args →
StorageBuffer-blocks transform is real engineering, not a cosmetic patch.

If the goal is to ship a working HIP-on-Vulkan path on LLVM 22 in the near
term, the existing `vulkanize/src/spirv_to_vulkan.cpp` post-processor
remains the realistic option. The clang-direct path is a longer-horizon
project (estimate: weeks, not days, even for a single kernel walking
skeleton) and would benefit from rebasing onto LLVM 23
(`chipstar-native-spirv`) where the in-tree SPIR-V Vulkan plumbing has had
another release of fixes.

### Minimal next-step proposal if pursued

1. Fix triple parsing so `spirv64-unknown-vulkan1.3-compute` is recognised
   (defects #2, #1, #8).
2. Add a `-mhip-output=vulkan` driver flag in HIPSPV that, when set, makes
   the frontend stamp `hlsl.shader("compute")` on every `__global__`
   function (defect #3) — gets us past the IRTranslator gate without any
   real ABI rewrite. **At that point Experiment B reproduces and we can
   measure how far an unwrapped HIP IR gets through the backend.**
3. Only after step 2 is in tree, tackle the pointer-arg → StorageBuffer
   transform (defects #5, #6, #7) — the heavy lift.

## Reproduction artefacts

Everything left in `/tmp` (transient — re-run from this doc's commands):

- `/tmp/poc_kernel.hip` — probe kernel
- `/tmp/poc_dev.ll` — Experiment A IR
- `/tmp/poc_minimal.ll` + `/tmp/poc_minimal.spv` — Experiment D
- `/tmp/poc-A.txt` … `/tmp/poc-D.txt`, `/tmp/poc-C.txt` … `/tmp/poc-C5.txt`
  — captured stderr per experiment

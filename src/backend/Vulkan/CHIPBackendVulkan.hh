/*
 * Copyright (c) 2021-26 chipStar developers
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

/**
 * @file CHIPBackendVulkan.hh
 * @brief Phase 2 header skeleton for the native Vulkan backend.
 *
 * This header declares every chipstar::* subclass the runtime needs to back
 * `CHIP_BE=vulkan`, with full member-variable layout, method signatures, and
 * the per-method ownership map for the Phase 3 parallel implementer fan-out.
 * All method bodies in the matching `.cc` are stubs that throw via `unimpl()`;
 * Phase 3 agents (I1..I12) fill them in.
 *
 * ============================================================================
 * Phase 3 ownership map (method group -> agent)
 * ============================================================================
 *
 * I1  Instance + Device bootstrap
 *     - CHIPBackendVulkan::initializeImpl
 *     - CHIPBackendVulkan::initializeFromNative
 *     - CHIPBackendVulkan::uninitialize
 *     - CHIPBackendVulkan::ctor / dtor (Instance + DebugMessenger lifetime)
 *     - CHIPDeviceVulkan::create
 *     - CHIPDeviceVulkan::populateDevicePropertiesImpl
 *     - CHIPDeviceVulkan::resetImpl
 *     - CHIPDeviceVulkan::createContext
 *     - Device-level Vulkan object lifetime (LogicalDevice_, Allocator_,
 *       CommandPool_, PipelineCache_, TimestampQueryPool_) ctor/dtor wiring
 *
 * I2  VMA memory allocator
 *     - CHIPContextVulkan::allocateImpl   (device/host/managed branches)
 *     - CHIPContextVulkan::freeImpl
 *     - CHIPContextVulkan::isAllocatedPtrMappedToVM
 *     - VMA initialization (VmaAllocatorCreateInfo) inside Device bootstrap
 *       hand-off (I1 calls into I2's allocator-construction helper)
 *     - VkBuffer lookup table (DevPtrToBuffer_) maintenance
 *
 * I3  Shader module + pipeline cache
 *     - CHIPModuleVulkan::compile
 *     - CHIPDeviceVulkan::compile (factory wrapper around CHIPModuleVulkan)
 *     - Per-(kernel, spec-constants) VkPipeline + VkDescriptorSetLayout +
 *       VkPipelineLayout cache management
 *     - SPV reflection -> KernelReflection conversion (consumes I/F2 parser)
 *
 * I4  Kernel + argument binding
 *     - CHIPKernelVulkan::getAttributes
 *     - CHIPExecItemVulkan::setupAllArgs
 *     - CHIPExecItemVulkan::clone
 *     - CHIPExecItemVulkan::setKernel / getKernel
 *     - Push-constant blob packing + descriptor-set update build
 *
 * I5  Event class
 *     - CHIPEventVulkan::ctor (timestamp slot allocation)
 *     - CHIPEventVulkan::wait
 *     - CHIPEventVulkan::updateFinishStatus
 *     - CHIPEventVulkan::getElapsedTime
 *     - CHIPEventVulkan::hostSignal
 *     - Timestamp scaling via VkPhysicalDeviceLimits::timestampPeriod
 *
 * I6  Queue lifecycle + barriers
 *     - CHIPQueueVulkan::ctor / dtor (per-queue command pool, primary buffer
 *       ring, current cmd-buffer state)
 *     - CHIPQueueVulkan::finish
 *     - CHIPQueueVulkan::query
 *     - CHIPQueueVulkan::recordEvent (vkCmdWriteTimestamp)
 *     - CHIPQueueVulkan::enqueueMarkerImpl
 *     - CHIPQueueVulkan::enqueueBarrierImpl
 *     - CHIPQueueVulkan::getBackendHandles
 *     - CHIPQueueVulkan::memPrefetchImpl (no-op for the spike)
 *
 * I7  launchImpl (the integration point)
 *     - CHIPQueueVulkan::launchImpl: pipeline bind + descriptor-set bind +
 *       push-constant write + vkCmdDispatch + submit-with-fence + recorded
 *       timestamp -> return shared event
 *
 * I8  Memory copy / fill family
 *     - CHIPQueueVulkan::memCopyAsyncImpl   (vkCmdCopyBuffer + staging)
 *     - CHIPQueueVulkan::memCopy2DAsyncImpl
 *     - CHIPQueueVulkan::memCopy3DAsyncImpl
 *     - CHIPQueueVulkan::memFillAsyncImpl   (vkCmdFillBuffer / vkCmdUpdateBuffer)
 *     - Staging-buffer pool management for H2D/D2H paths
 *
 * I9  EventMonitor (background fence-polling)
 *     - EventMonitorVulkan::monitor
 *
 * I10 Texture + graph stubs
 *     - CHIPDeviceVulkan::createTexture (returns nullptr / throws notSupported)
 *     - CHIPDeviceVulkan::destroyTexture
 *     - CHIPTextureVulkan::ctor / dtor
 *
 * I11 Backend factory methods
 *     - CHIPBackendVulkan::createExecItem
 *     - CHIPBackendVulkan::createCHIPQueue
 *     - CHIPBackendVulkan::createEventShared
 *     - CHIPBackendVulkan::createEvent
 *     - CHIPBackendVulkan::createCallbackData
 *     - CHIPBackendVulkan::createEventMonitor_
 *     - CHIPBackendVulkan::getDefaultJitFlags
 *     - CHIPBackendVulkan::ReqNumHandles
 *     - CHIPBackendVulkan::getHipEvent / getNativeEvent
 *     - CHIPDeviceVulkan::createQueue (delegates to backend factory)
 *
 * I12 Build + test harness
 *     - scripts/check.py: add `vulkan` backend arm
 *     - samples/CMakeLists.txt: enable samples under Vulkan backend
 *     - tests/known_failures.yaml: provisional entries for not-yet-passing
 *     (No code in this header is owned by I12; listed here for completeness.)
 *
 * ============================================================================
 * Spike scope reminders (per Phase-1 worklog)
 * ============================================================================
 * - SPV uses Logical GLSL450 addressing. Kernel args bind via StorageBuffer
 *   descriptors (set=0, binding=N) + ArgumentPodPushConstant push constants.
 *   `VK_KHR_buffer_device_address` is NOT used.
 * - Mandatory device feature: `shaderInt8`. Mandatory extension:
 *   `VK_KHR_shader_non_semantic_info`. Other capabilities (atomic_float,
 *   shader_clock, 16bit_storage, int64) are enabled per-module from the SPV
 *   `OpCapability` lines (I1 detects, I3 consumes).
 * - Textures + HIP graphs are stubbed as `hipErrorNotSupported` (I10).
 * - Single `CHIPBackendVulkan.cc` (no per-agent split): mirrors OpenCL/Level0
 *   structural pattern, sidesteps the symbol-visibility friction of split
 *   files, and Phase-3 conflicts resolve cleanly because each agent owns a
 *   disjoint banner-delimited section of the file.
 */

#ifndef CHIP_BACKEND_VULKAN_H
#define CHIP_BACKEND_VULKAN_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

// VMA is brought in via the `thirdparty/vma` PRIVATE include path on the CHIP
// target (gated on Vulkan_FOUND). Including the header here exposes its
// typedefs (VmaAllocator, VmaAllocation, VmaAllocationInfo) to consumers of
// this .hh file. The single `VMA_IMPLEMENTATION` translation-unit definition
// lives in `CHIPBackendVulkan.cc`.
#include "vk_mem_alloc.h"

#include "../../CHIPBackend.hh"
#include "../../SPVReflection.hh"

// ============================================================================
// Forward declarations
// ============================================================================
class CHIPBackendVulkan;
class CHIPContextVulkan;
class CHIPDeviceVulkan;
class CHIPQueueVulkan;
class CHIPModuleVulkan;
class CHIPKernelVulkan;
class CHIPExecItemVulkan;
class CHIPEventVulkan;
class CHIPTextureVulkan;
class EventMonitorVulkan;

// ============================================================================
// KernelReflection
// ============================================================================
//
// Per-kernel, post-reflection record carried on CHIPModuleVulkan and consumed
// at descriptor-set / push-constant binding time (I3 fills, I4/I7 consume).
//
// Field origins:
//   - StorageBufferArgs: `ArgumentStorageBuffer(kernel, ordinal, set, binding)`
//     ExtInst entries from `NonSemantic.ClspvReflection.5`.
//   - PushConstantArgs:  `ArgumentPodPushConstant(kernel, ordinal, offset, size)`
//     ExtInst entries (offset==Words[7], size==Words[8] in the parser).
//   - HiddenDGArgs:      bridging-pass-emitted hidden args mapping a kernel
//     ordinal to a `__device__` global symbol; bound from the per-symbol
//     buffer at launch time.
//   - PushConstantBlockSize: total bytes the push-constant block occupies
//     for this kernel; bounded by VkPhysicalDeviceLimits::maxPushConstantsSize
//     (Vulkan 1.2 spec-min is 128 B; the toolchain may emit up to 256 B).
// ============================================================================
struct VulkanStorageBufferArg {
  uint32_t Ordinal = 0;    ///< Kernel-arg index (post-bridging-pass).
  uint32_t Set = 0;        ///< Descriptor set (always 0 in the H4 path).
  uint32_t Binding = 0;    ///< Binding within the descriptor set.
};

struct VulkanPushConstantArg {
  uint32_t Ordinal = 0;    ///< Kernel-arg index.
  uint32_t Offset = 0;     ///< Byte offset within the push-constant block.
  uint32_t Size = 0;       ///< Size in bytes (4, 8, 16, ...).
};

struct VulkanKernelReflection {
  std::string Name;                              ///< Demangled kernel name.
  std::vector<VulkanStorageBufferArg> Buffers;   ///< set=0 bindings.
  std::vector<VulkanPushConstantArg>  PushConst; ///< Push-constant slots.
  std::vector<SPVKernelDeviceGlobalArg> HiddenDGArgs; ///< Per-kernel device-global binds.
  uint32_t PushConstantBlockSize = 0;            ///< Total bytes used by all PushConst entries (rounded up).
  uint32_t MaxDescriptorBinding = 0;             ///< Highest binding used in set=0.
};

// ============================================================================
// CHIPEventVulkan
// ============================================================================
//
// Vulkan host-visible event backed by a VkFence and a slot in the device-owned
// timestamp query pool. The fence drives `wait()`/`updateFinishStatus()`; the
// query pool slot drives `getElapsedTime()` (scaled by timestampPeriod). When
// the fence is unsignaled and no GPU work has been enqueued, EventStatus_
// reflects EVENT_STATUS_INIT and hostSignal() flips it to RECORDED.
// Owned by: I5.
// ============================================================================
class CHIPEventVulkan : public chipstar::Event {
  /// Host-visible synchronization primitive. Allocated lazily from the
  /// owning device's fence pool on first use; reset on event recycle.
  VkFence Fence_ = VK_NULL_HANDLE;

  /// Index into the device-wide TimestampQueryPool_ for the recorded
  /// hardware timestamp. -1 sentinel means "no timestamp recorded yet";
  /// applies to user events created via hipEventCreateWithFlags +
  /// hipEventDisableTiming.
  int32_t TimestampSlot_ = -1;

  /// Fallback host timestamp captured in updateFinishStatus() when the
  /// query pool slot is invalid (e.g. hipEventDisableTiming). Used by
  /// getElapsedTime to give a coarse measurement.
  uint64_t HostTimestamp_ = 0;

  /// Cached scaled timestamp (nanoseconds) once the query result has been
  /// pulled and multiplied by VkPhysicalDeviceLimits::timestampPeriod. Set
  /// to UINT64_MAX while the slot is still pending on the GPU.
  uint64_t Timestamp_ = UINT64_MAX;

public:
  CHIPEventVulkan(chipstar::Context *Ctx,
                  chipstar::EventFlags Flags = chipstar::EventFlags());
  virtual ~CHIPEventVulkan() override;

  // chipstar::Event pure virtuals (owned by I5).
  virtual bool updateFinishStatus(bool ThrowErrorIfNotReady = true) override;
  virtual bool wait() override;
  virtual float getElapsedTime(chipstar::Event *Other) override;
  virtual void hostSignal() override;

  // Vulkan-side accessors (consumed by I6 in Queue::recordEvent and by
  // CHIPBackendVulkan::getNativeEvent for HIP interop).
  VkFence getFence() const { return Fence_; }
  void setFence(VkFence F) { Fence_ = F; }
  int32_t getTimestampSlot() const { return TimestampSlot_; }
  void setTimestampSlot(int32_t Slot) { TimestampSlot_ = Slot; }
  uint64_t &getHostTimestamp() { return HostTimestamp_; }
  uint64_t &getTimestamp() { return Timestamp_; }
};

// ============================================================================
// EventMonitorVulkan
// ============================================================================
//
// Background polling thread that walks the backend's Events vector, calls
// updateFinishStatus() on each, and drains the callback queue when their
// dependencies are satisfied. Vulkan has no async notification primitive
// (the closest analogue is vkWaitForFences with a timeout), so this mirrors
// the Level0 pattern: a 200us sleep loop until the backend signals stop.
// Owned by: I9.
// ============================================================================
class EventMonitorVulkan : public chipstar::EventMonitor {
public:
  EventMonitorVulkan();
  virtual ~EventMonitorVulkan();
  virtual void monitor() override;
};

// ============================================================================
// CHIPModuleVulkan
// ============================================================================
//
// One Vulkan shader module per compiled SPVModule + per-kernel reflection
// records. Pipeline / descriptor-set-layout / pipeline-layout objects are
// cached on the module (one VkShaderModule covers all kernels in the SPV;
// per-kernel pipelines are built lazily on first launch).
// Owned by: I3.
// ============================================================================
class CHIPModuleVulkan : public chipstar::Module {
  /// Owning device for back-pointer to VkDevice and to the device-level
  /// pipeline cache. Set on compile().
  CHIPDeviceVulkan *ChipDevice_ = nullptr;

  /// The single VkShaderModule covering every entry point in this SPV.
  VkShaderModule ShaderModule_ = VK_NULL_HANDLE;

  /// Per-kernel reflection records, indexed by HostFName. Populated during
  /// compile() from `tryAnalyzeVulkanReflection`. Looked up by I4 to build
  /// descriptor-set updates and push-constant blobs.
  std::unordered_map<std::string, VulkanKernelReflection> Reflection_;

  /// Per-kernel descriptor-set-layout cache (one entry per kernel). The
  /// CHIPKernelVulkan instance references this via Module_; lifetime is
  /// tied to the module.
  std::unordered_map<std::string, VkDescriptorSetLayout> DSLayouts_;

  /// Per-kernel pipeline-layout cache (DSLayout + push-constant range).
  std::unordered_map<std::string, VkPipelineLayout> PipelineLayouts_;

  /// Per-(kernel, workgroup-size) compute pipeline cache. Key is
  /// "<HostFName>:WxHxD" where W/H/D are the SpecId 0/1/2 LocalSize spec
  /// constants. HIP's `blockDim` from the launch call becomes the Vulkan
  /// workgroup size via specialization at pipeline creation time.
  std::unordered_map<std::string, VkPipeline> Pipelines_;

  /// Per-symbol storage for `__device__` global variables. The bridging
  /// pass embeds these into the SPV via `__hipspv_dg_<sym>__sz_<n>__init_<hex>`
  /// OpName encodings and emits per-kernel "hidden" StorageBuffer args
  /// pointing at them. I3's compile() allocates one VkBuffer per global
  /// (initialized from `SPVDeviceGlobal::InitData`) and stores the handle
  /// here; I4's setupAllArgs looks them up by symbol name when binding the
  /// descriptor set.
  struct DeviceGlobalEntry {
    VkBuffer Buffer = VK_NULL_HANDLE;
    VmaAllocation Allocation = VK_NULL_HANDLE;
    size_t Size = 0;
  };
  std::unordered_map<std::string, DeviceGlobalEntry> DeviceGlobals_;

  /// Guards lazy compile() across threads when the FatBin module is reached
  /// for the first time on a freshly created queue.
  mutable std::mutex ModuleMtx_;

public:
  CHIPModuleVulkan(const SPVModule &SrcMod);
  virtual ~CHIPModuleVulkan() override;

  // chipstar::Module pure virtuals (owned by I3).
  virtual void compile(chipstar::Device *ChipDev) override;

  // Vulkan-side accessors used by I4 (kernel arg binding) and I7 (launchImpl).
  VkShaderModule getShaderModule() const { return ShaderModule_; }
  CHIPDeviceVulkan *getDevice() const { return ChipDevice_; }

  /// Look up a `__device__` global by symbol name. Returns nullptr if the
  /// symbol isn't backed by this module's SPV.
  const DeviceGlobalEntry *getDeviceGlobal(const std::string &Symbol) const {
    auto It = DeviceGlobals_.find(Symbol);
    return It == DeviceGlobals_.end() ? nullptr : &It->second;
  }

  /// Return reflection for a given kernel; nullptr if absent. I3 ensures
  /// every kernel that survives compile() has a record.
  const VulkanKernelReflection *getReflection(const std::string &Name) const;

  /// Return the cached descriptor-set layout for a kernel, building it on
  /// first request via the per-kernel reflection record.
  VkDescriptorSetLayout getOrCreateDescriptorSetLayout(const std::string &KernelName);

  /// Return the cached pipeline layout for a kernel (DSLayout +
  /// push-constant range), building it on first request.
  VkPipelineLayout getOrCreatePipelineLayout(const std::string &KernelName);

  /// Return the cached compute pipeline for a (kernel, workgroup-size)
  /// pair, building it on first request. The workgroup size is baked
  /// into the pipeline via VkSpecializationInfo (SpecId 0/1/2 = x/y/z),
  /// so a fresh pipeline is created per unique block dimension HIP
  /// launches at. Vulkan silently ignores spec data for SpecIds the SPV
  /// doesn't declare (multi-entry-point SPVs where inject_reflection.py
  /// only rewrote a subset's LocalSize to LocalSizeId).
  VkPipeline getOrCreatePipeline(const std::string &KernelName, dim3 BlockDim);
};

// ============================================================================
// CHIPKernelVulkan
// ============================================================================
//
// Thin wrapper around a kernel entry point inside a CHIPModuleVulkan. The
// expensive Vulkan objects (VkPipeline, VkDescriptorSetLayout,
// VkPipelineLayout) live on the parent module's cache; this class just keys
// into that cache by kernel name and exposes the per-kernel reflection.
// Owned by: I4.
// ============================================================================
class CHIPKernelVulkan : public chipstar::Kernel {
  /// Parent module (back-pointer, non-owning).
  CHIPModuleVulkan *Module_ = nullptr;

  /// Cached reference into Module_->Reflection_. Set on construction;
  /// nullptr if the kernel name was not present in the reflection (which
  /// would be a hard error caught in I3's compile()).
  const VulkanKernelReflection *Reflection_ = nullptr;

public:
  CHIPKernelVulkan(std::string HostFName, SPVFuncInfo *FuncInfo,
                   CHIPModuleVulkan *Parent);
  virtual ~CHIPKernelVulkan() override;

  // chipstar::Kernel pure virtuals.
  virtual hipError_t getAttributes(hipFuncAttributes *Attr) override;
  virtual chipstar::Module *getModule() override;
  virtual const chipstar::Module *getModule() const override;

  // Vulkan-side accessors (consumed by I4 in ExecItem::setupAllArgs and by
  // I7 in launchImpl for descriptor-set / push-constant / pipeline lookup).
  CHIPModuleVulkan *getVulkanModule() const { return Module_; }
  const VulkanKernelReflection *getReflection() const { return Reflection_; }
  void bindReflection(const VulkanKernelReflection *R) { Reflection_ = R; }
};

// ============================================================================
// CHIPExecItemVulkan
// ============================================================================
//
// Holds a packed push-constant blob and a flat list of VkBuffer handles to
// bind into the kernel's descriptor set at launch time. Built on every
// HIP launch via setupAllArgs() (I4) from the runtime's `Args_` array and
// the reflection on the kernel.
// Owned by: I4.
// ============================================================================
class CHIPExecItemVulkan : public chipstar::ExecItem {
  /// The kernel this exec item targets. Set via setKernel(); read by
  /// launchImpl() in I7.
  CHIPKernelVulkan *ChipKernel_ = nullptr;

  /// Host-side staging for the push-constant block. Sized to
  /// Reflection.PushConstantBlockSize on setKernel(); populated on
  /// setupAllArgs(); fed verbatim to vkCmdPushConstants on launch.
  std::vector<uint8_t> PushConstantBlob_;

  /// Per-storage-buffer-arg VkBuffer handles, indexed by descriptor binding
  /// number. Resolved from `Args_[i]` (a `void*` device pointer) by looking
  /// the pointer up in the context's DevPtrToBuffer_ map. Empty slots stay
  /// VK_NULL_HANDLE — I7 fails the launch if any slot is null at submit.
  std::vector<VkBuffer> BufferBindings_;

  /// Mirrors the descriptor sizes for vkUpdateDescriptorSets — Range[i] is
  /// the AllocationInfo::Size of BufferBindings_[i] or VK_WHOLE_SIZE.
  std::vector<VkDeviceSize> BufferRanges_;

  /// Per-binding byte offset into BufferBindings_[i]. Non-zero when the user
  /// passed a pointer into the middle of a hipMalloc'd region (e.g.
  /// `kernel<<<...>>>(&Hmm[k * NUM_ELMS], ...)` for managed memory). Driven
  /// by getDevPtrEntryContaining() in setupAllArgs and consumed by
  /// launchImpl's VkDescriptorBufferInfo::offset. Must respect the device's
  /// minStorageBufferOffsetAlignment limit; non-aligned offsets are rejected
  /// at setupAllArgs time.
  std::vector<VkDeviceSize> BufferOffsets_;
public:
  CHIPExecItemVulkan(dim3 GridDim, dim3 BlockDim, size_t SharedMem,
                     hipStream_t ChipQueue);
  CHIPExecItemVulkan(const CHIPExecItemVulkan &Other);
  virtual ~CHIPExecItemVulkan() override;

  // chipstar::ExecItem pure virtuals (owned by I4).
  virtual chipstar::ExecItem *clone() const override;
  virtual void setKernel(chipstar::Kernel *Kernel) override;
  virtual chipstar::Kernel *getKernel() override;
  virtual void setupAllArgs() override;

  // Vulkan-side accessors (consumed by I7 in launchImpl).
  CHIPKernelVulkan *getVulkanKernel() const { return ChipKernel_; }
  const std::vector<uint8_t> &getPushConstantBlob() const { return PushConstantBlob_; }
  const std::vector<VkBuffer> &getBufferBindings() const { return BufferBindings_; }
  const std::vector<VkDeviceSize> &getBufferRanges() const { return BufferRanges_; }
  const std::vector<VkDeviceSize> &getBufferOffsets() const { return BufferOffsets_; }
};

// ============================================================================
// CHIPContextVulkan
// ============================================================================
//
// 1:1 with a CHIPDeviceVulkan for the spike (multi-device contexts are out of
// scope). Owns the address->VkBuffer translation table that I7 consults to
// resolve kernel buffer arguments, and the AllocationTracker (inherited from
// chipstar::Device::AllocTracker — Context just bookkeeps).
// Owned by: I2.
// ============================================================================
class CHIPContextVulkan : public chipstar::Context {
  /// Address -> (VkBuffer, VmaAllocation, size) lookup for translating raw
  /// HIP device pointers (returned by hipMalloc) into the VkBuffer + offset
  /// pair vkCmdCopyBuffer / vkUpdateDescriptorSets requires.
  ///
  /// Populated by allocateImpl(), drained by freeImpl(). Held under the
  /// inherited ContextMtx.
  struct DevPtrEntry {
    VkBuffer Buffer = VK_NULL_HANDLE;
    VmaAllocation Allocation = VK_NULL_HANDLE;
    VmaAllocationInfo AllocInfo{};
    size_t Size = 0;
    hipMemoryType MemType = hipMemoryTypeDevice;
    chipstar::HostAllocFlags Flags;
  };
  std::unordered_map<const void *, DevPtrEntry> DevPtrToEntry_;

public:
  CHIPContextVulkan();
  virtual ~CHIPContextVulkan() override;

  // chipstar::Context pure virtuals (owned by I2).
  virtual void *
  allocateImpl(size_t Size, size_t Alignment, hipMemoryType MemType,
               chipstar::HostAllocFlags Flags = chipstar::HostAllocFlags())
      override;
  virtual bool isAllocatedPtrMappedToVM(void *Ptr) override;
  virtual void freeImpl(void *Ptr) override;

  // Vulkan-side accessors (consumed by I7 launchImpl + I8 memcpy paths).
  /// Look up the VkBuffer backing a raw device pointer. Returns VK_NULL_HANDLE
  /// if the pointer was not allocated through this context.
  VkBuffer translateDevPtrToBuffer(const void *DevPtr) const;
  /// Look up the full DevPtrEntry (buffer, size, allocation) for a pointer.
  /// Returns nullptr if not found.
  const DevPtrEntry *getDevPtrEntry(const void *DevPtr) const;

  /// Convenience: return the device this context is bound to as the
  /// concrete CHIPDeviceVulkan*. Sugar over chipstar::Context::getDevice().
  CHIPDeviceVulkan *getVulkanDevice() const;
};

// ============================================================================
// CHIPDeviceVulkan
// ============================================================================
//
// Top-of-Vulkan ownership: holds the VkPhysicalDevice + VkDevice, the
// VmaAllocator, the command pool used for transient command buffers, the
// device-wide pipeline cache, and the timestamp query pool driving HIP
// event timing. Created via the static `create()` factory so the
// chipstar::Device::init() two-phase initialization can run before any
// virtual call.
// Owned by: I1 (members + ctor/dtor); I3 owns compile() factory.
// ============================================================================
class CHIPDeviceVulkan : public chipstar::Device {
  // ----- Vulkan handles (I1 lifetime) -----
  /// The physical device chosen by Backend::initializeImpl() based on
  /// feature-gating (mandatory: shaderInt8). Non-owning — owned by Instance.
  VkPhysicalDevice PhysicalDevice_ = VK_NULL_HANDLE;

  /// The logical device created from PhysicalDevice_ with the chosen queue
  /// family. Destroyed in ~CHIPDeviceVulkan(); must outlive every other
  /// member declared below.
  VkDevice LogicalDevice_ = VK_NULL_HANDLE;

  /// Compute queue from the chosen queue family (single queue for the spike;
  /// multi-queue dispatch is a Phase-5 follow-up).
  VkQueue ComputeQueue_ = VK_NULL_HANDLE;
  uint32_t ComputeQueueFamilyIndex_ = ~0u;

  /// Cached physical-device properties / features / subgroup properties for
  /// populateDevicePropertiesImpl(). SubgroupProperties_.subgroupSize is
  /// what we map HIP warpSize to.
  VkPhysicalDeviceProperties Properties_{};
  VkPhysicalDeviceFeatures Features_{};
  VkPhysicalDeviceSubgroupProperties SubgroupProperties_{};

  // ----- VMA + shared device-level Vulkan objects -----
  /// VMA allocator instance. Constructed alongside LogicalDevice_, destroyed
  /// before it. Used by every CHIPContextVulkan::allocateImpl call.
  VmaAllocator Allocator_ = VK_NULL_HANDLE;

  /// Backend-wide pipeline cache. Persisted to disk across runs is a
  /// Phase-5 polish item; for the spike it is in-memory only.
  VkPipelineCache PipelineCache_ = VK_NULL_HANDLE;

  /// Transient-command-buffer pool used by queues for one-shot command
  /// buffers (memcpy staging, marker timestamps). Queues that need their
  /// own pool (e.g. for queue-family-thread affinity) allocate their own.
  VkCommandPool CommandPool_ = VK_NULL_HANDLE;
  std::mutex CommandPoolMtx_;

  /// Device-wide timestamp query pool. Sized to TimestampPoolSize_ slots;
  /// each CHIPEventVulkan grabs one slot. Slots are tracked in a free list
  /// held under TimestampMtx_.
  VkQueryPool TimestampQueryPool_ = VK_NULL_HANDLE;
  static constexpr uint32_t TimestampPoolSize_ = 4096;
  std::vector<int32_t> TimestampFreeList_;
  std::mutex TimestampMtx_;

  /// VkFence pool. Recycled across CHIPEventVulkan lifetimes to avoid the
  /// per-event allocation cost. Reset to unsignaled state on return.
  std::vector<VkFence> FencePool_;
  std::mutex FencePoolMtx_;

  /// Per-Vulkan-spec: vkQueueSubmit on a VkQueue must be externally
  /// synchronized. Because every CHIPQueueVulkan stream on this device shares
  /// the single ComputeQueue_, all submits are serialized through this mutex.
  /// Mirrors the I6 spike-grade "one VkQueue per device" model; lifts to a
  /// per-VkQueue mutex when multi-queue support lands.
  mutable std::mutex SubmitMtx_;

  // ----- Per-module feature flags -----
  /// Set on Device construction from feature-detection; consumed by I3
  /// when validating that a module's required capabilities are satisfied.
  bool HasShaderInt8_ = false;
  bool HasShaderInt64_ = false;
  bool HasShader16BitStorage_ = false;
  bool HasShaderClock_ = false;
  bool HasShaderAtomicFloat_ = false;

  // Only constructable through `create()`; mirrors the OpenCL/Level0 pattern
  // so chipstar::Device::init()'s virtual dispatch into populateDevicePropertiesImpl()
  // sees a fully-constructed concrete object.
  CHIPDeviceVulkan(CHIPContextVulkan *ChipContext, VkPhysicalDevice PhysDev,
                   int Idx);

public:
  virtual ~CHIPDeviceVulkan() override;

  static CHIPDeviceVulkan *create(CHIPContextVulkan *ChipContext,
                                  VkPhysicalDevice PhysDev, int Idx);

  // chipstar::Device pure virtuals.
  virtual chipstar::Context *createContext() override;        // I1
  virtual void populateDevicePropertiesImpl() override;       // I1
  virtual chipstar::Queue *createQueue(chipstar::QueueFlags Flags,
                                       int Priority) override; // I11
  virtual chipstar::Queue *createQueue(const uintptr_t *NativeHandles,
                                       int NumHandles) override; // I11
  virtual chipstar::Texture *
  createTexture(const hipResourceDesc *ResDesc, const hipTextureDesc *TexDesc,
                const struct hipResourceViewDesc *ResViewDesc) override; // I10
  virtual void destroyTexture(chipstar::Texture *TextureObject) override; // I10
  virtual void resetImpl() override;                          // I1
  virtual chipstar::Module *compile(const SPVModule &Src) override; // I3

  // ----- Vulkan-side accessors (consumed broadly across Phase 3) -----
  VkPhysicalDevice getPhysicalDevice() const { return PhysicalDevice_; }
  VkDevice getLogicalDevice() const { return LogicalDevice_; }
  VkQueue getComputeQueue() const { return ComputeQueue_; }
  uint32_t getComputeQueueFamilyIndex() const { return ComputeQueueFamilyIndex_; }
  const VkPhysicalDeviceProperties &getProperties() const { return Properties_; }
  const VkPhysicalDeviceFeatures &getFeatures() const { return Features_; }
  const VkPhysicalDeviceSubgroupProperties &getSubgroupProperties() const {
    return SubgroupProperties_;
  }
  VmaAllocator getAllocator() const { return Allocator_; }
  std::mutex &getSubmitMtx() const { return SubmitMtx_; }
  VkPipelineCache getPipelineCache() const { return PipelineCache_; }
  VkCommandPool getCommandPool() const { return CommandPool_; }
  VkQueryPool getTimestampQueryPool() const { return TimestampQueryPool_; }

  bool hasShaderInt8() const { return HasShaderInt8_; }
  bool hasShaderInt64() const { return HasShaderInt64_; }
  bool hasShader16BitStorage() const { return HasShader16BitStorage_; }
  bool hasShaderClock() const { return HasShaderClock_; }
  bool hasShaderAtomicFloat() const { return HasShaderAtomicFloat_; }

  // ----- Pool helpers (owned by I1 / I5 / I6) -----
  /// Acquire a fence from the recycle pool, creating a new one if empty.
  VkFence acquireFence();                                     // I1
  /// Return a fence to the recycle pool. Resets the fence to unsignaled.
  void releaseFence(VkFence F);                               // I1
  /// Acquire a timestamp slot; returns -1 if the pool is exhausted (caller
  /// falls back to host timing).
  int32_t acquireTimestampSlot();                             // I5
  void releaseTimestampSlot(int32_t Slot);                    // I5

  CHIPContextVulkan *getContext() override {
    return static_cast<CHIPContextVulkan *>(this->Device::getContext());
  }
};

// ============================================================================
// CHIPQueueVulkan
// ============================================================================
//
// Wraps the shared VkQueue (since the spike uses a single queue family / one
// queue) with a per-queue command pool and a primary command-buffer ring.
// Each enqueue records into the "current" command buffer; submit happens on
// barrier, finish, or when the ring rolls over. A timeline semaphore
// (TimelineSemaphore_) drives in-queue ordering across submits without the
// overhead of per-submit fences.
// Owned by: I6 (ctor/dtor + lifecycle); I7 (launchImpl); I8 (mem* paths).
// ============================================================================
class CHIPQueueVulkan : public chipstar::Queue {
  /// Back-pointer to the concrete device for VkDevice access.
  CHIPDeviceVulkan *ChipDevice_ = nullptr;

  /// Per-queue command pool. Marked TRANSIENT so the runtime can reset and
  /// reuse buffers cheaply. RESET_COMMAND_BUFFER_BIT lets us free
  /// individual buffers via vkResetCommandBuffer (used after fence-completes
  /// the corresponding submit).
  VkCommandPool CommandPool_ = VK_NULL_HANDLE;

  /// Ring of primary command buffers. Each enqueue (launchImpl, memCopy,
  /// memFill, marker, barrier) takes the "current" buffer; on submit the
  /// ring slot rolls forward. CmdBufferRing_.size() == RingCapacity_ on
  /// initialization; capacity is chosen as a heuristic (16 for the spike).
  static constexpr uint32_t RingCapacity_ = 16;
  std::vector<VkCommandBuffer> CmdBufferRing_;
  uint32_t RingHead_ = 0;

  /// Timeline semaphore monotonically incremented on every submit; used by
  /// enqueueBarrierImpl to chain dependencies across submits without
  /// host-side waits.
  VkSemaphore TimelineSemaphore_ = VK_NULL_HANDLE;
  uint64_t TimelineValue_ = 0;

  /// Fence used by finish() / query(). Reset and re-signaled on every
  /// submit boundary; vkGetFenceStatus drives query(), vkWaitForFences
  /// drives finish().
  VkFence FinishFence_ = VK_NULL_HANDLE;

  /// Track if any work has been submitted on this queue. Mirrors the
  /// OpenCL backend's IsEmptyQueue_ usage so the default-stream
  /// cross-queue sync helper in chipstar::Queue can skip empty queues.
  std::atomic<bool> IsEmptyQueue_{true};

  /// Cross-queue dependency markers held alive so checkEvents() can't
  /// recycle them while this queue still references them as wait deps
  /// (mirrors the Level0 pattern in addDependenciesQueueSyncImpl).
  std::vector<std::shared_ptr<chipstar::Event>> CrossQueueDeps_;
  std::mutex CrossQueueDepsMtx_;

  /// Guards Ring/Pool/Semaphore mutation. Coarse-grained for the spike;
  /// can be split if profiling shows contention.
  std::mutex QueueOpMtx_;

protected:
  virtual void storeCrossQueueDeps(
      std::vector<std::shared_ptr<chipstar::Event>> Markers) override;

public:
  CHIPQueueVulkan() = delete;
  CHIPQueueVulkan(const CHIPQueueVulkan &) = delete;
  CHIPQueueVulkan(chipstar::Device *ChipDevice, int Priority);
  virtual ~CHIPQueueVulkan() override;

  // chipstar::Queue pure virtuals.
  virtual void recordEvent(chipstar::Event *Event) override;          // I6
  virtual bool isEmptyQueue() override { return IsEmptyQueue_.load(); }
  virtual std::shared_ptr<chipstar::Event>
  memCopyAsyncImpl(void *Dst, const void *Src, size_t Size,
                   hipMemcpyKind Kind) override;                      // I8
  virtual std::shared_ptr<chipstar::Event>
  memFillAsyncImpl(void *Dst, size_t Size, const void *Pattern,
                   size_t PatternSize) override;                      // I8
  virtual std::shared_ptr<chipstar::Event>
  memCopy2DAsyncImpl(void *Dst, size_t DPitch, const void *Src, size_t SPitch,
                     size_t Width, size_t Height, hipMemcpyKind Kind)
      override;                                                       // I8
  virtual std::shared_ptr<chipstar::Event>
  memCopy3DAsyncImpl(void *Dst, size_t DPitch, size_t DSPitch, const void *Src,
                     size_t SPitch, size_t SSPitch, size_t Width, size_t Height,
                     size_t Depth, hipMemcpyKind Kind) override;      // I8
  virtual std::shared_ptr<chipstar::Event>
  launchImpl(chipstar::ExecItem *ExecItem) override;                  // I7
  virtual void finish() override;                                     // I6
  virtual bool query() override;                                      // I6
  virtual std::shared_ptr<chipstar::Event> enqueueBarrierImpl(
      const std::vector<std::shared_ptr<chipstar::Event>> &EventsToWaitFor)
      override;                                                       // I6
  virtual std::shared_ptr<chipstar::Event> enqueueMarkerImpl() override; // I6
  virtual std::shared_ptr<chipstar::Event>
  memPrefetchImpl(const void *Ptr, size_t Count, int DstDevId) override; // I6
  virtual hipError_t getBackendHandles(uintptr_t *NativeHandles,
                                       int *NumHandles) override;     // I6

  // Vulkan-side accessors.
  CHIPDeviceVulkan *getVulkanDevice() const { return ChipDevice_; }
  VkCommandPool getCommandPool() const { return CommandPool_; }
  VkSemaphore getTimelineSemaphore() const { return TimelineSemaphore_; }
  uint64_t getTimelineValue() const { return TimelineValue_; }

  /// Acquire a command buffer from the ring for recording. Called by I7/I8
  /// to begin a recording session. Caller is responsible for vkBeginCommandBuffer.
  VkCommandBuffer acquireCmdBuffer();                                 // I6

  /// Submit the given command buffer with `EventsToWaitFor` as the wait list
  /// and the queue's TimelineSemaphore_ + a freshly-acquired event-fence as
  /// the signal pair. Returns the shared event (already track()ed by the
  /// backend). Used by every enqueue path.
  std::shared_ptr<chipstar::Event> submitWithEvent(
      VkCommandBuffer Buf,
      const std::vector<std::shared_ptr<chipstar::Event>> &EventsToWaitFor);   // I6

  CHIPContextVulkan *getContext() override {
    return static_cast<CHIPContextVulkan *>(ChipContext_);
  }
};

// ============================================================================
// CHIPBackendVulkan
// ============================================================================
//
// Owns the VkInstance, the (optional) VkDebugUtilsMessengerEXT, the list of
// CHIPDeviceVulkan instances, and the singleton EventMonitorVulkan. The
// CHIP runtime instantiates this via the factory in src/CHIPDriver.cc when
// the user sets `CHIP_BE=vulkan`.
// Owned by: I1 (instance + uninitialize); I11 (factory methods).
// ============================================================================
class CHIPBackendVulkan : public chipstar::Backend {
  /// The single VkInstance shared by every CHIPDeviceVulkan. Lifetime: full
  /// process. Created in initializeImpl(), destroyed in uninitialize().
  VkInstance Instance_ = VK_NULL_HANDLE;

  /// Debug-utils messenger when CHIP_VK_VALIDATION=ON (cmake option). Optional;
  /// remains VK_NULL_HANDLE when validation is disabled or the layer isn't
  /// present.
  VkDebugUtilsMessengerEXT DebugMessenger_ = VK_NULL_HANDLE;

  /// True if the VK_LAYER_KHRONOS_validation layer was enabled at instance
  /// creation. Logged at startup for triage; consumed by I1.
  bool ValidationEnabled_ = false;

  /// True if the VK_EXT_debug_utils instance extension was enabled. Consumed
  /// by I1 when wiring up DebugMessenger_.
  bool DebugUtilsEnabled_ = false;

  /// Cached list of extensions actually enabled on Instance_ — used by I1
  /// for instance-extension feature gating at device-creation time.
  std::vector<std::string> EnabledInstanceExtensions_;

public:
  CHIPBackendVulkan();
  virtual ~CHIPBackendVulkan() override;

  // chipstar::Backend pure virtuals.
  virtual chipstar::ExecItem *createExecItem(dim3 GridDim, dim3 BlockDim,
                                             size_t SharedMem,
                                             hipStream_t ChipQueue) override; // I11
  virtual std::string getDefaultJitFlags() override;                          // I11
  virtual int ReqNumHandles() override;                                       // I11
  virtual void initializeImpl() override;                                     // I1
  virtual void initializeFromNative(const uintptr_t *NativeHandles,
                                    int NumHandles) override;                 // I1
  virtual void uninitialize() override;                                       // I1
  virtual chipstar::Queue *createCHIPQueue(chipstar::Device *ChipDev) override; // I11
  virtual std::shared_ptr<chipstar::Event>
  createEventShared(chipstar::Context *ChipCtx, chipstar::EventFlags Flags,
                    std::string Msg) override;                                // I11
  virtual chipstar::Event *
  createEvent(chipstar::Context *ChipCtx,
              chipstar::EventFlags Flags = chipstar::EventFlags()) override; // I11
  virtual chipstar::CallbackData *
  createCallbackData(hipStreamCallback_t Callback, void *UserData,
                     chipstar::Queue *ChipQ) override;                       // I11
  virtual chipstar::EventMonitor *createEventMonitor_() override;            // I11
  virtual hipEvent_t getHipEvent(void *NativeEvent) override;                // I11
  virtual void *getNativeEvent(hipEvent_t HipEvent) override;                // I11

  // Vulkan-side accessors.
  VkInstance getInstance() const { return Instance_; }
  bool isValidationEnabled() const { return ValidationEnabled_; }
  bool isDebugUtilsEnabled() const { return DebugUtilsEnabled_; }
  const std::vector<std::string> &getEnabledInstanceExtensions() const {
    return EnabledInstanceExtensions_;
  }
};

// ============================================================================
// CHIPTextureVulkan
// ============================================================================
//
// Stub. Texture support is out of scope for the spike — every code path that
// would land here is expected to surface `hipErrorNotSupported` at the HIP
// API boundary before construction is attempted. Definition kept so the
// virtual hierarchy resolves and so a future Phase-5 implementation drops in
// without an ABI change.
// Owned by: I10.
// ============================================================================
class CHIPTextureVulkan : public chipstar::Texture {
public:
  CHIPTextureVulkan() = delete;
  explicit CHIPTextureVulkan(const hipResourceDesc &ResDesc);
  virtual ~CHIPTextureVulkan() override;
};

#endif // CHIP_BACKEND_VULKAN_H

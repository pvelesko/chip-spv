# Vulkan Memory Allocator (VMA) — vendored

This directory vendors the single-file, header-only [Vulkan Memory
Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
library from AMD GPUOpen. It is consumed by the `CHIPBackendVulkan` backend
(`src/backend/Vulkan/`) to manage `VkDeviceMemory` suballocation, host/device
mapping, and `hipMallocManaged` semantics on top of raw Vulkan 1.2.

## Source

- Upstream: <https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator>
- Header pulled from:
  `https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/master/include/vk_mem_alloc.h`
- License text pulled from:
  `https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/master/LICENSE.txt`

## Version pinned

- Branch: `master`
- Commit: `b3cbbb43ea3a506dffe10759e205a41c27c35ae2` (2026-03-02, merge of
  upstream PR #526 "New updates to Vulkan Memory Allocator")
- This is the tip of `master` at the time of vendoring. The most recent
  tagged release is `v3.3.0`; `master` is one merge commit ahead. Pinning to
  the SHA above is what should be reproduced if this header is refreshed.

## License

MIT (see `LICENSE.txt` in this directory). VMA is © 2017-2026 Advanced Micro
Devices, Inc. and licensed under the MIT License.

## Files

- `vk_mem_alloc.h` — the entire VMA library (~19,875 lines, ~750 KB),
  header-only.
- `LICENSE.txt` — MIT license text from upstream.
- `README.md` — this file.

## Usage (consumer notes for Phase 2/3 of the Vulkan backend)

VMA is a single-file header-only library. Include the header anywhere you
need its declarations:

```cpp
#include "vk_mem_alloc.h"
```

In **exactly one** translation unit, define `VMA_IMPLEMENTATION` *before*
the include to emit the function definitions:

```cpp
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
```

Recommendation: place the `VMA_IMPLEMENTATION` define + include in
`src/backend/Vulkan/CHIPBackendVulkan.cc` (the Vulkan backend's primary
translation unit, owned by Phase 2 / agent H1). Do **not** define
`VMA_IMPLEMENTATION` in headers or in more than one `.cc` file — that will
cause multiply-defined-symbol link errors.

CMake-side, the include path `thirdparty/vma` is added to the Vulkan
backend target's `target_include_directories` by agent F1 (Phase 1).

## Refreshing this vendored copy

To update to a newer upstream version:

```bash
curl -L -o thirdparty/vma/vk_mem_alloc.h \
  https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/<sha-or-tag>/include/vk_mem_alloc.h
curl -L -o thirdparty/vma/LICENSE.txt \
  https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/<sha-or-tag>/LICENSE.txt
```

Then update the "Version pinned" section above with the new commit SHA and
date, and re-run the Vulkan backend test suite to confirm no API breakage.

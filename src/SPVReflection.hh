/*
 * Copyright (c) 2024-26 chipStar developers
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

/// \file SPVReflection.hh
/// Shared Vulkan-flavored SPIR-V reflection parser.
///
/// Parses NonSemantic.ClspvReflection.5 ExtInst metadata emitted by our
/// HIPSPV bridging pass + inject_reflection.py post-processor into the
/// chipStar runtime `SPVModuleInfo` shape (kernels, args, push-constant
/// layouts, `__device__` globals).
///
/// Both runtime paths consume the same reflection:
///   - CHIP_BE=opencl  → CHIPBackendOpenCL  → clvk (clCreateProgramWithBinary)
///   - CHIP_BE=vulkan  → CHIPBackendVulkan  → direct VkComputePipeline
///
/// Keeping the parser shared avoids reflection-schema drift between the
/// two backends.

#ifndef CHIPSTAR_SRC_SPV_REFLECTION_HH
#define CHIPSTAR_SRC_SPV_REFLECTION_HH

#include "common.hh"

#include <cstddef>
#include <cstdint>

/// Try to interpret `Stream`/`NumWords` as Vulkan-flavored SPIR-V (Logical or
/// PhysicalStorageBuffer64 addressing, NonSemantic.ClspvReflection.5 metadata)
/// and populate `Output`. Returns true on a successful Vulkan-reflection
/// parse; false to indicate the caller should fall through to the legacy
/// OpenCL-flavored parser.
bool tryAnalyzeVulkanReflection(const uint32_t *Stream, size_t NumWords,
                                SPVModuleInfo &Output);

#endif // CHIPSTAR_SRC_SPV_REFLECTION_HH

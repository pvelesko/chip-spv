/*
Copyright (c) 2025 chipStar developers.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef SRC_SPIRV_TO_VULKAN_HH
#define SRC_SPIRV_TO_VULKAN_HH

#include <cstdint>
#include <string>
#include <vector>

/// Convert OpenCL SPIR-V (Physical64 addressing, Kernel execution model) into
/// Vulkan SPIR-V (Logical GLSL450, GLCompute execution model, StorageBuffer
/// descriptors, push constants, and embedded NonSemantic.ClspvReflection
/// metadata).  The transform is performed entirely in memory without invoking
/// any external tools.  The result is ready to be passed directly to
/// clCreateProgramWithIL; clvk detects the GLCompute execution model and
/// bypasses its own clspv compilation pipeline.
///
/// @param opencl_spv     Raw bytes of the OpenCL SPIR-V module.
/// @param clspv_options  Extra options forwarded to clspv (e.g.
///                       "-physical-storage-buffers").
/// @param error_msg      If non-null and the transform fails, receives a
///                       human-readable description of the error.
/// @return               SPIR-V words of the Vulkan module, or empty on error.
std::vector<uint32_t> openclToVulkanSpirv(
    const std::vector<uint8_t>& opencl_spv,
    const std::string& clspv_options = "",
    std::string* error_msg = nullptr);

#endif // SRC_SPIRV_TO_VULKAN_HH

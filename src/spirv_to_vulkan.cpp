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

/// Convert OpenCL SPIR-V → Vulkan SPIR-V via llvm-spirv + clspv subprocesses.
/// The output Vulkan SPIR-V (GLCompute, StorageBuffer, NonSemantic.ClspvReflection)
/// can be passed directly to clCreateProgramWithIL; clvk will detect the
/// GLCompute execution model and skip its own clspv compilation pipeline.

#include "spirv_to_vulkan.hh"
#include "Utils.hh"
#include "logging.hh"
#include "chipStarConfig.hh"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

/// Write binary data to a file.  Returns false on error.
static bool writeBinaryToFile(const fs::path &Path, const uint8_t *Data,
                               size_t Size) {
  std::ofstream F(Path, std::ios::binary);
  if (!F.is_open())
    return false;
  F.write(reinterpret_cast<const char *>(Data), Size);
  return F.good();
}

/// Read all bytes of a binary file.  Returns empty vector on error.
static std::vector<uint32_t> readSpirvFile(const fs::path &Path) {
  std::ifstream F(Path, std::ios::binary | std::ios::ate);
  if (!F.is_open())
    return {};
  auto Size = F.tellg();
  if (Size <= 0 || (Size % 4) != 0)
    return {};
  F.seekg(0);
  std::vector<uint32_t> Words(Size / 4);
  F.read(reinterpret_cast<char *>(Words.data()), Size);
  return F.good() ? Words : std::vector<uint32_t>{};
}

/// Run a shell command, redirecting stdout+stderr to LogFile.
/// Returns true if the command exits with status 0.
static bool runCommand(const std::string &Cmd, const fs::path &LogFile) {
  std::string Full = Cmd + " >" + LogFile.string() + " 2>&1";
  logDebug("spirv_to_vulkan: running: {}", Cmd);
  int Ret = std::system(Full.c_str());
  if (Ret != 0) {
    // Dump log to debug output so it's visible with CHIP_LOGLEVEL=debug.
    if (auto Contents = readFromFile(LogFile))
      logDebug("spirv_to_vulkan cmd log:\n{}", *Contents);
  }
  return Ret == 0;
}

std::vector<uint32_t> openclToVulkanSpirv(const std::vector<uint8_t> &opencl_spv,
                                           const std::string &clspv_options,
                                           std::string *error_msg) {
  auto SetErr = [&](const std::string &Msg) -> std::vector<uint32_t> {
    logError("openclToVulkanSpirv: {}", Msg);
    if (error_msg)
      *error_msg = Msg;
    return {};
  };

  // Resolve tool paths -------------------------------------------------------

#if defined(LLVM_TOOLS_BINARY_DIR)
  const std::string LlvmSpirv = std::string(LLVM_TOOLS_BINARY_DIR) + "/llvm-spirv";
#else
  const std::string LlvmSpirv = "llvm-spirv";
#endif

#if defined(CHIP_CLSPV_BIN)
  const std::string Clspv = (!std::string(CHIP_CLSPV_BIN).empty())
                                 ? std::string(CHIP_CLSPV_BIN)
                                 : std::string("clspv");
#else
  const std::string Clspv = "clspv";
#endif

  // Create temp directory ----------------------------------------------------

  auto TmpDirOpt = createTemporaryDirectory();
  if (!TmpDirOpt)
    return SetErr("Could not create temporary directory for OpenCL→Vulkan SPIR-V conversion");

  fs::path TmpDir = *TmpDirOpt;
  fs::path SpvIn   = TmpDir / "source.spv";
  fs::path BcFile  = TmpDir / "source.bc";
  fs::path SpvOut  = TmpDir / "output.spv";
  fs::path Log1    = TmpDir / "llvmspirv.log";
  fs::path Log2    = TmpDir / "clspv.log";

  // Write OpenCL SPIR-V to disk ----------------------------------------------

  if (!writeBinaryToFile(SpvIn, opencl_spv.data(), opencl_spv.size()))
    return SetErr("Could not write OpenCL SPIR-V to temp file: " + SpvIn.string());

  // Step 1: llvm-spirv -r  (SPIR-V → LLVM bitcode) --------------------------

  std::string Cmd1 = "\"" + LlvmSpirv + "\" -r"
                     " -o \"" + BcFile.string() + "\""
                     " \"" + SpvIn.string() + "\"";

  if (!runCommand(Cmd1, Log1)) {
    std::string Log;
    if (auto C = readFromFile(Log1))
      Log = *C;
    return SetErr("llvm-spirv -r failed:\n" + Log);
  }

  // Step 2: clspv  (LLVM bitcode → Vulkan SPIR-V) ----------------------------

  // Build the base clspv flags:
  //   -x ir                         : bitcode input
  //   -lower-generic-address-space  : required for chipStar generic AS usage
  //   -inline-entry-points          : required for CL3.0 / generic AS kernels
  //   --arch spir64                 : required when -physical-storage-buffers
  //                                   is specified (64-bit physical pointers)
  //
  // The caller supplies any additional flags (e.g. -physical-storage-buffers).
  bool HasPSB = clspv_options.find("-physical-storage-buffers") != std::string::npos;
  std::string ArchFlag = HasPSB ? " --arch spir64" : "";

  auto MakeCmd2 = [&](const std::string &Opts) -> std::string {
    return "\"" + Clspv + "\""
           " -x ir"
           " -lower-generic-address-space"
           " -inline-entry-points" +
           ArchFlag +
           " " + Opts +
           " -o \"" + SpvOut.string() + "\""
           " \"" + BcFile.string() + "\"";
  };

  if (!runCommand(MakeCmd2(clspv_options), Log2)) {
    std::string Log;
    if (auto C = readFromFile(Log2))
      Log = *C;

    // Retry without -physical-storage-buffers if it was included and failed.
    // clspv's PhysicalPointerArgsPass can crash on certain kernel patterns.
    if (HasPSB) {
      logWarn("openclToVulkanSpirv: clspv failed with -physical-storage-buffers, "
              "retrying without it");
      std::string Opts2 = clspv_options;
      auto Pos = Opts2.find("-physical-storage-buffers");
      Opts2.erase(Pos, std::string("-physical-storage-buffers").size());
      ArchFlag = ""; // no longer need spir64 without PSB

      if (!runCommand(MakeCmd2(Opts2), Log2)) {
        if (auto C = readFromFile(Log2))
          Log = *C;
        fs::remove_all(TmpDir);
        return SetErr("clspv failed (with and without -physical-storage-buffers):\n" + Log);
      }
    } else {
      fs::remove_all(TmpDir);
      return SetErr("clspv failed:\n" + Log);
    }
  }

  // Read back Vulkan SPIR-V --------------------------------------------------

  auto Words = readSpirvFile(SpvOut);
  fs::remove_all(TmpDir);

  if (Words.empty())
    return SetErr("Could not read clspv output from: " + SpvOut.string());

  logDebug("openclToVulkanSpirv: produced {} words of Vulkan SPIR-V", Words.size());
  return Words;
}

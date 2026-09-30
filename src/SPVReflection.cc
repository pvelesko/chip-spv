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

#include "SPVReflection.hh"
#include "logging.hh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cxxabi.h>
#include <map>
#include <string>
#include <vector>

using InstWord = uint32_t;

// Helpers live in an anonymous namespace; only `tryAnalyzeVulkanReflection`
// is part of the public API exposed via SPVReflection.hh.
namespace {

// Parse the demangled signature from `cxxabi.h` to recover, for
// each HIP source argument, whether the type is a pointer/reference.
// Used to remap clspv-flavored reflection arg ordinals (which place
// pointers/storage-buffers before PODs/push-constants) back into the
// original HIP source argument order — required so that the HIP
// runtime, which always passes its kernel-argument list in HIP
// source order, hands the right value to the right OpenCL kernel
// arg index.
//
// On any failure (unrecognised mangling, demangler unavailable,
// etc.) returns an empty vector and the caller falls back to
// treating reflection ord as the HIP source order — which preserves
// the legacy behaviour for kernels that happen to have all
// pointers, all PODs, or only one argument.

// Walk a demangled C++ signature `Name(Params)` and return a vector
// of bools over the top-level parameters: true = pointer/reference.
// Skips templates, parens, brackets, and angle brackets when
// counting commas at depth 0.
static std::vector<bool>
classifyDemangledArgs(const std::string &Demangled) {
  std::vector<bool> Result;
  // The parameter list is the paren group closing the signature, e.g.
  // `(anonymous namespace)::f<int>(int const*, long)`: scan back from the end.
  size_t RParen = Demangled.rfind(')');
  if (RParen == std::string::npos)
    return Result;
  size_t LParen = std::string::npos;
  int PDepth = 0;
  for (size_t I = RParen + 1; I-- > 0;) {
    if (Demangled[I] == ')')
      PDepth++;
    else if (Demangled[I] == '(' && --PDepth == 0) {
      LParen = I;
      break;
    }
  }
  if (LParen == std::string::npos)
    return Result;
  // Split the parameter list on commas at depth 0.
  std::string ParamList = Demangled.substr(LParen + 1, RParen - LParen - 1);
  size_t Start = 0;
  int Depth = 0;
  auto pushParam = [&](size_t Begin, size_t End) {
    // Trim trailing whitespace.
    while (End > Begin && std::isspace(static_cast<unsigned char>(ParamList[End - 1])))
      End--;
    while (Begin < End && std::isspace(static_cast<unsigned char>(ParamList[Begin])))
      Begin++;
    if (Begin >= End)
      return;
    // Top-level pointer/reference: the *last* non-whitespace, non-cv-token
    // character is '*' or '&'.
    bool IsPtr = false;
    // Look at the trailing token, skipping cv qualifiers.
    std::string_view Sv(ParamList.data() + Begin, End - Begin);
    // Find rightmost '*' or '&' that is at top level.
    int InnerDepth = 0;
    for (size_t I = Sv.size(); I-- > 0;) {
      char C = Sv[I];
      if (C == '>' || C == ')' || C == ']') InnerDepth++;
      else if (C == '<' || C == '(' || C == '[') InnerDepth--;
      else if (InnerDepth == 0 && (C == '*' || C == '&')) {
        IsPtr = true;
        break;
      } else if (InnerDepth == 0 && !std::isspace(static_cast<unsigned char>(C))) {
        // Hit an alphanumeric/identifier character → not a pointer.
        if (std::isalnum(static_cast<unsigned char>(C)) || C == '_') {
          // Could still be a const/volatile suffix — keep looking
          // past 'const' / 'volatile' tokens.
          continue;
        }
      }
    }
    Result.push_back(IsPtr);
  };
  for (size_t I = 0; I < ParamList.size(); ++I) {
    char C = ParamList[I];
    if (C == '<' || C == '(' || C == '[')
      Depth++;
    else if (C == '>' || C == ')' || C == ']')
      Depth--;
    else if (C == ',' && Depth == 0) {
      pushParam(Start, I);
      Start = I + 1;
    }
  }
  pushParam(Start, ParamList.size());
  // Special case: single "void" parameter means no args.
  if (Result.size() == 1) {
    // Trim ParamList and check.
    std::string T = ParamList;
    T.erase(std::remove_if(T.begin(), T.end(),
                           [](char C) {
                             return std::isspace(static_cast<unsigned char>(C));
                           }),
            T.end());
    if (T == "void")
      Result.clear();
  }
  return Result;
}

static std::vector<bool>
classifyHipKernelArgsByMangling(const std::string &MangledName) {
  std::vector<bool> Empty;
  int Status = 0;
  char *Demangled =
      abi::__cxa_demangle(MangledName.c_str(), nullptr, nullptr, &Status);
  if (Status != 0 || !Demangled) {
    if (Demangled)
      free(Demangled);
    return Empty;
  }
  std::string D(Demangled);
  free(Demangled);
  return classifyDemangledArgs(D);
}
} // namespace

bool tryAnalyzeVulkanReflection(const InstWord *Stream, size_t NumWords,
                                SPVModuleInfo &Output) {
  // Find OpMemoryModel; accept either Logical addressing (Vulkan default) or
  // PhysicalStorageBuffer64 (used by HIPSPV's bridging pass for byte-strided
  // pitched-pointer access via BDA). Both are Vulkan-flavored SPV with the
  // same ClspvReflection-style kernel metadata.
  bool IsVulkanAddressing = false;
  size_t I = 5;
  while (I < NumWords) {
    InstWord W = Stream[I];
    uint16_t Wc = (W >> 16) & 0xFFFF;
    uint16_t Op = W & 0xFFFF;
    if (Wc == 0)
      return false;
    if (Op == 14 /*OpMemoryModel*/) {
      if (Wc >= 3) {
        InstWord AM = Stream[I + 1];
        // 0 = Logical, 5348 = PhysicalStorageBuffer64.
        if (AM == 0 || AM == 5348)
          IsVulkanAddressing = true;
      }
      break;
    }
    I += Wc;
  }
  if (!IsVulkanAddressing)
    return false;

  // Pass 1: gather OpExtInstImport (find ClspvReflection set), OpString,
  //         OpConstant uint values, and OpEntryPoint (function-id → name).
  std::map<InstWord, std::string> Strings;
  std::map<InstWord, uint64_t> Consts;
  std::map<InstWord, std::string> EntryPointNames; // fn_id → entry name
  std::map<InstWord, std::vector<InstWord>> EntryPointIface; // fn_id → vars
  // Phase G4: per-result-id OpName plus DescriptorSet/Binding decorations,
  // and result-ids of OpVariable in StorageBuffer storage class. These three
  // sources together identify the device-global descriptors emitted by the
  // HIPSPVLowerToHLSLShape bridging pass at (set=1, binding=N).
  std::map<InstWord, std::string> ResultNames;       // id → OpName
  std::map<InstWord, uint32_t> DescSet;              // id → DescriptorSet
  std::map<InstWord, uint32_t> DescBinding;          // id → Binding
  std::set<InstWord> StorageBufferVars;              // result-ids
  InstWord ReflSetId = 0;

  auto decodeOpString = [&](const InstWord *Words, size_t WC) -> std::string {
    // OpString result_id <literal string>
    // Operand layout: word[1] = result, words[2..] = string
    std::string S;
    for (size_t k = 2; k < WC; ++k) {
      InstWord W = Words[k];
      for (int b = 0; b < 4; ++b) {
        char C = static_cast<char>((W >> (8 * b)) & 0xff);
        if (C == 0)
          return S;
        S.push_back(C);
      }
    }
    return S;
  };

  I = 5;
  while (I < NumWords) {
    InstWord W = Stream[I];
    uint16_t Wc = (W >> 16) & 0xFFFF;
    uint16_t Op = W & 0xFFFF;
    if (Wc == 0)
      return false;
    const InstWord *Words = &Stream[I];
    if (Op == 11 /*OpExtInstImport*/ && Wc >= 3) {
      InstWord Id = Words[1];
      // The remainder is a literal string with the set name.
      std::string Name;
      for (size_t k = 2; k < Wc; ++k) {
        InstWord X = Words[k];
        bool Done = false;
        for (int b = 0; b < 4; ++b) {
          char C = static_cast<char>((X >> (8 * b)) & 0xff);
          if (C == 0) {
            Done = true;
            break;
          }
          Name.push_back(C);
        }
        if (Done)
          break;
      }
      if (Name.find("NonSemantic.ClspvReflection") != std::string::npos)
        ReflSetId = Id;
    } else if (Op == 7 /*OpString*/ && Wc >= 2) {
      Strings[Words[1]] = decodeOpString(Words, Wc);
    } else if (Op == 50 /*OpSpecConstant*/ && Wc >= 3) {
      // Skip — workgroup spec constants only.
    } else if (Op == 43 /*OpConstant*/ && Wc >= 4) {
      Consts[Words[2]] = static_cast<uint64_t>(Words[3]);
    } else if (Op == 15 /*OpEntryPoint*/ && Wc >= 4) {
      // Words[0]=opcode, [1]=execution_model, [2]=fn_id, [3..]=name+iface.
      InstWord FnId = Words[2];
      std::string EpName;
      size_t k = 3;
      for (; k < Wc; ++k) {
        InstWord X = Words[k];
        bool Done = false;
        for (int b = 0; b < 4; ++b) {
          char C = static_cast<char>((X >> (8 * b)) & 0xff);
          if (C == 0) {
            Done = true;
            break;
          }
          EpName.push_back(C);
        }
        if (Done)
          break;
      }
      EntryPointNames[FnId] = EpName;
      EntryPointIface[FnId].assign(Words + std::min<size_t>(k + 1, Wc),
                                   Words + Wc);
    } else if (Op == 5 /*OpName*/ && Wc >= 3) {
      // OpName target_id <literal name>
      InstWord Tgt = Words[1];
      std::string N;
      for (size_t k = 2; k < Wc; ++k) {
        InstWord X = Words[k];
        bool Done = false;
        for (int b = 0; b < 4; ++b) {
          char C = static_cast<char>((X >> (8 * b)) & 0xff);
          if (C == 0) {
            Done = true;
            break;
          }
          N.push_back(C);
        }
        if (Done)
          break;
      }
      ResultNames[Tgt] = std::move(N);
    } else if (Op == 71 /*OpDecorate*/ && Wc >= 4) {
      // OpDecorate target_id Decoration <literals>
      InstWord Tgt = Words[1];
      uint32_t Dec = Words[2];
      if (Dec == 34 /*DescriptorSet*/ && Wc >= 4)
        DescSet[Tgt] = Words[3];
      else if (Dec == 33 /*Binding*/ && Wc >= 4)
        DescBinding[Tgt] = Words[3];
    } else if (Op == 59 /*OpVariable*/ && Wc >= 4) {
      // OpVariable result_type result_id storage_class [initializer]
      InstWord Sc = Words[3];
      if (Sc == 12 /*StorageBuffer*/)
        StorageBufferVars.insert(Words[2]);
    }
    I += Wc;
  }

  // Phase H4: gather set=0 StorageBuffer descriptors whose OpName matches
  // the bridging-pass naming convention `__hipspv_dg_<symbol>`. Each such
  // descriptor is a hidden kernel arg synthesised by
  // lowerDeviceGlobalsToStorageBuffers; the runtime needs to bind the
  // chipStar-allocated cl_mem of `<symbol>` at the corresponding kernel
  // arg ordinal. Build a per-binding -> symbol map here; per-kernel ord
  // mapping is built below in the reflection-walk.
  std::map<uint32_t, std::string> BindingToDGSymbol; // set=0 binding -> symbol
  std::map<InstWord, std::string> VarToDGSymbol;
  {
    static constexpr const char *kPrefix = "__hipspv_dg_";
    static constexpr size_t kPrefixLen = 12;
    std::set<std::string> SeenSymbols;
    auto hexNibble = [](char C) -> int {
      if (C >= '0' && C <= '9')
        return C - '0';
      if (C >= 'a' && C <= 'f')
        return 10 + (C - 'a');
      if (C >= 'A' && C <= 'F')
        return 10 + (C - 'A');
      return -1;
    };
    for (InstWord Id : StorageBufferVars) {
      auto SI = DescSet.find(Id);
      auto BI = DescBinding.find(Id);
      auto NI = ResultNames.find(Id);
      if (SI == DescSet.end() || BI == DescBinding.end() ||
          NI == ResultNames.end())
        continue;
      if (SI->second != 0)
        continue;
      const std::string &Full = NI->second;
      if (Full.compare(0, kPrefixLen, kPrefix) != 0)
        continue;
      // Phase I3: split off the optional `__sz_<size>__init_<hex>` suffix
      // emitted by HIPSPVLowerToHLSLShape::encodeDgArgName so the runtime
      // can size + seed each device global's storage buffer at allocation
      // time and re-seed it on hipDeviceReset.
      std::string Tail = Full.substr(kPrefixLen);
      std::string Symbol = Tail;
      size_t SizeBytes = 0;
      std::vector<uint8_t> InitBytes;
      auto SzPos = Tail.find("__sz_");
      if (SzPos != std::string::npos) {
        Symbol = Tail.substr(0, SzPos);
        std::string Rest = Tail.substr(SzPos + 5);
        size_t SzEnd = Rest.find("__init_");
        std::string SizeStr =
            (SzEnd == std::string::npos) ? Rest : Rest.substr(0, SzEnd);
        try {
          SizeBytes = static_cast<size_t>(std::stoull(SizeStr));
        } catch (...) {
          SizeBytes = 0;
        }
        if (SzEnd != std::string::npos) {
          std::string Hex = Rest.substr(SzEnd + 7);
          size_t HexLen = 0;
          while (HexLen < Hex.size() && hexNibble(Hex[HexLen]) >= 0)
            ++HexLen;
          HexLen &= ~size_t(1); // round down to even
          InitBytes.reserve(HexLen / 2);
          for (size_t k = 0; k + 1 < HexLen; k += 2) {
            int Hi = hexNibble(Hex[k]);
            int Lo = hexNibble(Hex[k + 1]);
            if (Hi < 0 || Lo < 0)
              break;
            InitBytes.push_back(static_cast<uint8_t>((Hi << 4) | Lo));
          }
        }
      }
      if (Symbol.empty())
        continue;
      BindingToDGSymbol[BI->second] = Symbol;
      VarToDGSymbol[Id] = Symbol;
      if (SeenSymbols.insert(Symbol).second) {
        SPVDeviceGlobal DG;
        DG.Name = Symbol;
        DG.Set = 0;
        DG.Binding = BI->second;
        DG.Size = SizeBytes; // 0 for legacy SPVs without the __sz_ suffix.
        DG.InitData = std::move(InitBytes);
        Output.DeviceGlobals.push_back(std::move(DG));
      }
    }
  }

  if (ReflSetId == 0)
    return false; // No reflection — bail; caller will retry the OCL path.

  // Pass 2: walk OpExtInst calls into the reflection set. Build per-kernel
  // argument vectors indexed by the reflection Kernel result-id.
  struct KernelRecord {
    InstWord FunctionId = 0;
    std::string Name;
    std::vector<std::pair<uint32_t /*ord*/, SPVArgTypeInfo>> Args;
  };
  std::map<InstWord, KernelRecord> Kernels; // kernel_result_id → record

  I = 5;
  while (I < NumWords) {
    InstWord W = Stream[I];
    uint16_t Wc = (W >> 16) & 0xFFFF;
    uint16_t Op = W & 0xFFFF;
    if (Wc == 0)
      return false;
    const InstWord *Words = &Stream[I];
    if (Op == 12 /*OpExtInst*/ && Wc >= 5) {
      // [1]=result_type, [2]=result_id, [3]=ext_set, [4]=ext_op,
      // [5..]=operands.
      if (Words[3] == ReflSetId) {
        InstWord ResId = Words[2];
        InstWord ExtOp = Words[4];
        switch (ExtOp) {
        case 1: { // Kernel(fn_id, name_str, num_args, flags, attrs)
          if (Wc < 10)
            break;
          KernelRecord R;
          R.FunctionId = Words[5];
          auto NameIt = Strings.find(Words[6]);
          if (NameIt != Strings.end())
            R.Name = NameIt->second;
          else
            R.Name = EntryPointNames.count(R.FunctionId)
                         ? EntryPointNames[R.FunctionId]
                         : std::string{};
          Kernels[ResId] = std::move(R);
          break;
        }
        case 3: { // ArgumentStorageBuffer(kernel, ord, set, binding)
          if (Wc < 9)
            break;
          auto It = Kernels.find(Words[5]);
          if (It == Kernels.end())
            break;
          uint32_t Ord = static_cast<uint32_t>(Consts[Words[6]]);
          uint32_t ArgSet = static_cast<uint32_t>(Consts[Words[7]]);
          uint32_t ArgBinding = static_cast<uint32_t>(Consts[Words[8]]);
          SPVArgTypeInfo Ti;
          Ti.Kind = SPVTypeKind::Pointer;
          Ti.StorageClass = SPVStorageClass::CrossWorkgroup;
          Ti.Size = 8; // pointer size
          Ti.Binding = static_cast<int>(ArgBinding);
          // Phase H4: if this descriptor is a hidden device-global arg,
          // record (kernel, ord, symbol) so the launch path can bind the
          // chipStar-allocated cl_mem, and mark its kind so visitClientArgs
          // skips it.
          if (ArgSet == 0) {
            // Bindings are per kernel, so resolve through this entry
            // point's interface; other kernels may reuse the binding.
            const std::string *Sym = nullptr;
            auto IfIt = EntryPointIface.find(It->second.FunctionId);
            if (IfIt != EntryPointIface.end()) {
              for (InstWord V : IfIt->second) {
                auto VS = VarToDGSymbol.find(V);
                auto VB = DescBinding.find(V);
                if (VS != VarToDGSymbol.end() && VB != DescBinding.end() &&
                    VB->second == ArgBinding && DescSet[V] == 0) {
                  Sym = &VS->second;
                  break;
                }
              }
            } else {
              auto SymIt = BindingToDGSymbol.find(ArgBinding);
              if (SymIt != BindingToDGSymbol.end())
                Sym = &SymIt->second;
            }
            if (Sym) {
              SPVKernelDeviceGlobalArg HA;
              HA.Symbol = *Sym;
              HA.ArgIndex = Ord;
              Output.HiddenDGArgsByKernel[It->second.Name].push_back(HA);
              Ti.Kind = SPVTypeKind::DeviceGlobalHidden;
              logDebug(
                  "Reflect kernel='{}' hidden DG arg ord={} symbol='{}'",
                  It->second.Name, Ord, *Sym);
            }
          }
          It->second.Args.emplace_back(Ord, Ti);
          logDebug("Reflect kernel='{}' ArgStorageBuffer ord={}",
                   It->second.Name, Ord);
          break;
        }
        case 5: { // ArgumentPodStorageBuffer(kernel, ord, set, binding,
                  //                          offset, size)
          if (Wc < 11)
            break;
          auto It = Kernels.find(Words[5]);
          if (It == Kernels.end())
            break;
          SPVArgTypeInfo Ti;
          Ti.Kind = SPVTypeKind::POD;
          Ti.StorageClass = SPVStorageClass::Private;
          Ti.Binding = static_cast<int>(Consts[Words[8]]);
          Ti.PushConstOffset = static_cast<int>(Consts[Words[9]]);
          Ti.Size = static_cast<size_t>(Consts[Words[10]]);
          It->second.Args.emplace_back(
              static_cast<uint32_t>(Consts[Words[6]]), Ti);
          break;
        }
        case 7: { // ArgumentPodPushConstant(kernel, ord, offset, size)
          if (Wc < 9)
            break;
          auto It = Kernels.find(Words[5]);
          if (It == Kernels.end())
            break;
          uint32_t Ord = static_cast<uint32_t>(Consts[Words[6]]);
          uint64_t Sz = Consts[Words[8]];
          SPVArgTypeInfo Ti;
          Ti.Kind = SPVTypeKind::POD;
          Ti.StorageClass = SPVStorageClass::Private;
          Ti.Size = static_cast<size_t>(Sz);
          Ti.PushConstOffset = static_cast<int>(Consts[Words[7]]);
          It->second.Args.emplace_back(Ord, Ti);
          logDebug("Reflect kernel='{}' ArgPodPushConstant ord={} size={}",
                   It->second.Name, Ord, Sz);
          break;
        }
        case 26: { // Phase Z3: ArgumentPointerPushConstant(kernel, ord, offset, size, [arg_info])
          // Buffer Device Address slot: 8-byte ulong at `offset` within the PC
          // block where the runtime must substitute the user's HIP pointer
          // with vkGetBufferDeviceAddress(buffer_lookup(pointer)). Multiple
          // slots may exist per kernel (e.g. a byval struct with several
          // pointer fields). The slot is NOT a separate ClientArg — it lives
          // inside an existing POD/byval kernel arg's push-constant region.
          if (Wc < 9)
            break;
          auto It = Kernels.find(Words[5]);
          if (It == Kernels.end())
            break;
          uint32_t Offset = static_cast<uint32_t>(Consts[Words[7]]);
          Output.BDAPointerSlotOffsetsByKernel[It->second.Name].push_back(Offset);
          logDebug("Reflect kernel='{}' ArgPointerPushConstant offset={}",
                   It->second.Name, Offset);
          break;
        }
        case 1000: { // chipStar: ArgumentPointerNullFlag(kernel, ord, offset)
          if (Wc < 8)
            break;
          auto It = Kernels.find(Words[5]);
          if (It == Kernels.end())
            break;
          Output.NullFlagSlotsByKernel[It->second.Name].emplace_back(
              static_cast<uint32_t>(Consts[Words[6]]),
              static_cast<uint32_t>(Consts[Words[7]]));
          break;
        }
        default:
          break;
        }
      }
    }
    I += Wc;
  }

  if (Kernels.empty())
    return false;

  // Build SPVFuncInfo entries keyed by kernel name.
  for (auto &Kv : Kernels) {
    auto &Rec = Kv.second;
    if (Rec.Name.empty())
      continue;
    // Sort args by clspv reflection ordinal. After this, the vector is
    // in OpenCL kernel signature order: storage buffers (user pointers,
    // then any hidden device-global descriptors) first, then push
    // constants (PODs).
    std::sort(Rec.Args.begin(), Rec.Args.end(),
              [](const auto &A, const auto &B) { return A.first < B.first; });

    // The HIP runtime walks ArgTypeInfo_ alongside ClientArgList in HIP
    // source order. The chipstarvulkan-flavored SPIR-V reorders args
    // (pointers/SBs first, PODs last), so we must rebuild the vector in
    // HIP source order with each entry's KernelArgIndex set to the OCL
    // kernel-arg ordinal (the reflection ord).
    auto IsPtrPerHipArg = classifyHipKernelArgsByMangling(Rec.Name);
    // The bridging pass recorded this kernel's source argument order, so the
    // ordinals already are HIP source positions.
    std::string SrcOrderMarker = ".hipspv.pcsargs." + Rec.Name;
    bool SourceOrdered = std::any_of(
        ResultNames.begin(), ResultNames.end(), [&](const auto &KV) {
          const std::string &N = KV.second;
          return N.compare(0, SrcOrderMarker.size(), SrcOrderMarker) == 0 &&
                 (N.size() == SrcOrderMarker.size() ||
                  N[SrcOrderMarker.size()] == '.');
        });

    // Partition the reflected args into three buckets.
    std::vector<size_t> PointerArgsOcl;
    std::vector<size_t> HiddenDgArgsOcl;
    std::vector<size_t> PodArgsOcl;
    for (size_t I = 0; I < Rec.Args.size(); ++I) {
      auto &Ti = Rec.Args[I].second;
      if (Ti.Kind == SPVTypeKind::Pointer)
        PointerArgsOcl.push_back(I);
      else if (Ti.Kind == SPVTypeKind::DeviceGlobalHidden)
        HiddenDgArgsOcl.push_back(I);
      else
        PodArgsOcl.push_back(I);
    }

    // Heuristic fallback for extern "C" kernels: when demangling produces no
    // signature (extern "C" symbols have unmangled names and so __cxa_demangle
    // returns nothing useful), we still need a HIP source-order arg-kind
    // sequence. Otherwise the legacy fallback emits reflection-ord order
    // (storage buffers first, then push constants), which mismatches HIP
    // source order whenever the kernel has both pointer and POD args — the
    // runtime then calls clSetKernelArgDevicePointerEXT on POD values and
    // launchImpl fails with CL_INVALID_KERNEL_ARGS.
    //
    // Assume the libCEED / common-HIP convention: PODs (scalars, sizes,
    // flags) come first, then output/input pointers. This matches every
    // extern "C" kernel exposed by libCEED's hip-ref/hip-shared backends
    // (Interp, Grad, Weight, *Transpose, *AtPoints, restriction kernels,
    // etc.) and is the dominant pattern in HIP-tests as well. Kernels
    // following a different convention will still misbehave under this
    // fallback — for those a proper SPV-side source-order encoding is
    // needed (see TODO below).
    //
    // TODO: encode HIP source arg-kinds directly in the SPV via the
    // HIPSPV bridging pass (e.g. as a private global string consumed by
    // hipspv-inject-reflection and surfaced through the kernel attrs
    // OpString), so this heuristic isn't needed.
    if (IsPtrPerHipArg.empty() && !SourceOrdered && !PointerArgsOcl.empty() &&
        !PodArgsOcl.empty()) {
      IsPtrPerHipArg.reserve(PodArgsOcl.size() + PointerArgsOcl.size());
      IsPtrPerHipArg.insert(IsPtrPerHipArg.end(), PodArgsOcl.size(), false);
      IsPtrPerHipArg.insert(IsPtrPerHipArg.end(), PointerArgsOcl.size(), true);
      logDebug("Reflect kernel='{}' demangle failed (extern \"C\"?); "
               "assuming source order = {} PODs then {} pointers",
               Rec.Name, PodArgsOcl.size(), PointerArgsOcl.size());
    }

    std::vector<SPVArgTypeInfo> ArgInfos;
    ArgInfos.reserve(Rec.Args.size());

    long NumHipPtrs =
        std::count(IsPtrPerHipArg.begin(), IsPtrPerHipArg.end(), true);
    long NumHipPods = (long)IsPtrPerHipArg.size() - NumHipPtrs;
    // Allow more reflected ords than HIP source args: chipstarvulkan
    // / clspv emits extra SBs (one per module-level descriptor) and
    // extra PCs across all kernels in the program. We bind only the
    // first NumHipPtrs SBs (in clspv ord order) to the HIP pointer
    // args and only the first NumHipPods PCs to the HIP POD args.
    // No reflected PODs at all but HIP has non-pointer args: they were all
    // unused (e.g. empty functors) and the backend dropped them.
    bool AllPodsDropped = PodArgsOcl.empty() && NumHipPods > 0;
    // Unused pointer args are dropped too; the pass numbers bindings in HIP
    // pointer order, so the survivors can still be placed by binding.
    bool PtrsByBinding = NumHipPtrs > (long)PointerArgsOcl.size() &&
                         std::all_of(PointerArgsOcl.begin(),
                                     PointerArgsOcl.end(), [&](size_t I) {
                                       int B = Rec.Args[I].second.Binding;
                                       return B >= 0 && B < NumHipPtrs;
                                     });
    bool RemapOk = !IsPtrPerHipArg.empty() &&
                   (NumHipPtrs <= (long)PointerArgsOcl.size() ||
                    PtrsByBinding) &&
                   (NumHipPods <= (long)PodArgsOcl.size() || AllPodsDropped);

    if (RemapOk) {
      size_t NextPtr = 0, NextPod = 0;
      for (bool IsPtr : IsPtrPerHipArg) {
        if (IsPtr && PtrsByBinding) {
          int Want = (int)NextPtr++;
          auto Hit = std::find_if(
              PointerArgsOcl.begin(), PointerArgsOcl.end(),
              [&](size_t I) { return Rec.Args[I].second.Binding == Want; });
          if (Hit == PointerArgsOcl.end()) {
            SPVArgTypeInfo Ti{};
            Ti.Kind = SPVTypeKind::POD;
            Ti.StorageClass = SPVStorageClass::Private;
            Ti.Size = 0;
            ArgInfos.push_back(Ti);
          } else {
            SPVArgTypeInfo Ti = Rec.Args[*Hit].second;
            Ti.KernelArgIndex = (int)Rec.Args[*Hit].first;
            ArgInfos.push_back(Ti);
          }
          continue;
        }
        if (!IsPtr && AllPodsDropped) {
          SPVArgTypeInfo Ti{};
          Ti.Kind = SPVTypeKind::POD;
          Ti.StorageClass = SPVStorageClass::Private;
          Ti.Size = 0;
          ArgInfos.push_back(Ti);
          continue;
        }
        size_t OclIdx;
        if (IsPtr)
          OclIdx = PointerArgsOcl[NextPtr++];
        else
          OclIdx = PodArgsOcl[NextPod++];
        SPVArgTypeInfo Ti = Rec.Args[OclIdx].second;
        Ti.KernelArgIndex = (int)Rec.Args[OclIdx].first;
        ArgInfos.push_back(Ti);
      }
      // Append hidden device-global descriptors and any leftover
      // pointer/POD ords that the program-wide reflection injected
      // across kernels (clspv exposes every module-level descriptor
      // as a kernel arg of every kernel). These don't consume a
      // ClientArgList slot. Pointer leftovers are reclassified as
      // DeviceGlobalHidden (with no associated symbol) so the OpenCL
      // launch path binds them to nullptr; POD leftovers are dropped
      // because clvk doesn't enforce push-constant binding count.
      for (size_t Idx : HiddenDgArgsOcl) {
        SPVArgTypeInfo Ti = Rec.Args[Idx].second;
        Ti.KernelArgIndex = (int)Rec.Args[Idx].first;
        ArgInfos.push_back(Ti);
      }
      for (size_t I = PtrsByBinding ? PointerArgsOcl.size() : NextPtr;
           I < PointerArgsOcl.size(); ++I) {
        SPVArgTypeInfo Ti = Rec.Args[PointerArgsOcl[I]].second;
        Ti.Kind = SPVTypeKind::DeviceGlobalHidden;
        Ti.KernelArgIndex = (int)Rec.Args[PointerArgsOcl[I]].first;
        ArgInfos.push_back(Ti);
      }
      logDebug(
          "Reflect kernel='{}' remapped {} HIP-source args (KernelArgIndex "
          "set per-arg, {} unbound SBs)",
          Rec.Name, IsPtrPerHipArg.size(),
          PointerArgsOcl.size() - NumHipPtrs);
    } else {
      // Fallback: keep the legacy behaviour of using reflection-ord
      // order. Works correctly for all-pointer or all-POD kernels.
      for (auto &P : Rec.Args)
        ArgInfos.push_back(P.second);
      logDebug("Reflect kernel='{}' using legacy ord order ({} args, "
               "demangle={} produced {} entries)",
               Rec.Name, Rec.Args.size(),
               IsPtrPerHipArg.empty() ? "failed" : "ok",
               IsPtrPerHipArg.size());
    }

    auto FInfo = std::make_shared<SPVFuncInfo>(ArgInfos);
    Output.FuncInfoMap.emplace(Rec.Name, std::move(FInfo));
  }

  Output.HasNoIGBAs = true; // Vulkan SPV doesn't use indirect global buffers.
  return !Output.FuncInfoMap.empty();
}

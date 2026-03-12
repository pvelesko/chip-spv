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

/// Pure in-memory SPIR-V transform: OpenCL Physical64 → Vulkan PSB SPIR-V.
/// No subprocess invocations. Only handles -physical-storage-buffers mode;
/// returns empty vector for non-PSB so callers fall back to clvk's pipeline.

#include "spirv_to_vulkan.hh"
#include "logging.hh"
#include "spirv.hh"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// SPIR-V opcode constants (raw uint32 values)
// ---------------------------------------------------------------------------
namespace Op {
  static const uint32_t Nop                     = 0;
  static const uint32_t Source                  = 3;
  static const uint32_t Name                    = 5;
  static const uint32_t MemberName              = 6;
  static const uint32_t String                  = 7;
  static const uint32_t ExtInstImport           = 11;
  static const uint32_t ExtInst                 = 12;
  static const uint32_t MemoryModel             = 14;
  static const uint32_t EntryPoint              = 15;
  static const uint32_t ExecutionMode           = 16;
  static const uint32_t ExecutionModeId         = 331;
  static const uint32_t Capability              = 17;
  static const uint32_t TypeVoid                = 19;
  static const uint32_t TypeInt                 = 21;
  static const uint32_t TypeFloat               = 22;
  static const uint32_t TypeVector              = 23;
  static const uint32_t TypeArray               = 28;
  static const uint32_t TypeStruct              = 30;
  static const uint32_t TypePointer             = 32;
  static const uint32_t TypeFunction            = 33;
  static const uint32_t Constant                = 43;
  static const uint32_t ConstantComposite       = 44;
  static const uint32_t SpecConstant            = 50;
  static const uint32_t SpecConstantComposite   = 51;
  static const uint32_t Function                = 54;
  static const uint32_t FunctionParameter       = 55;
  static const uint32_t FunctionEnd             = 56;
  static const uint32_t FunctionCall            = 57;
  static const uint32_t Variable                = 59;
  static const uint32_t Load                    = 61;
  static const uint32_t Store                   = 62;
  static const uint32_t AccessChain             = 65;
  static const uint32_t InBoundsAccessChain     = 66;
  static const uint32_t PtrAccessChain          = 67;
  static const uint32_t InBoundsPtrAccessChain  = 70;
  static const uint32_t Decorate                = 71;
  static const uint32_t MemberDecorate          = 72;
  static const uint32_t DecorationGroup         = 73;
  static const uint32_t GroupDecorate           = 74;
  static const uint32_t GroupMemberDecorate     = 75;
  static const uint32_t ConvertPtrToU           = 117;
  static const uint32_t PtrCastToGeneric        = 121;
  static const uint32_t GenericCastToPtr        = 122;
  static const uint32_t ConvertUToPtr           = 120;
  static const uint32_t Bitcast                 = 124;
  static const uint32_t CopyMemorySized         = 64;
  static const uint32_t Label                   = 248;
  static const uint32_t Branch                  = 249;
  static const uint32_t BranchConditional       = 250;
  static const uint32_t Switch                  = 251;
  static const uint32_t CompositeExtract         = 81;
  static const uint32_t UConvert                 = 113;
  static const uint32_t ControlBarrier           = 224;
  static const uint32_t MemoryBarrier            = 225;
  static const uint32_t Return                  = 253;
  static const uint32_t ReturnValue             = 254;
  static const uint32_t Unreachable             = 255;
  static const uint32_t LoopMerge              = 246;
  static const uint32_t SelectionMerge         = 247;
  static const uint32_t Extension               = 10;
}

// Storage classes
namespace SC {
  static const uint32_t UniformConstant    = 0;
  static const uint32_t Input              = 1;
  static const uint32_t Uniform            = 2;
  static const uint32_t Output             = 3;
  static const uint32_t WorkgroupLocal     = 4;
  static const uint32_t CrossWorkgroup     = 5;
  static const uint32_t Private           = 6;
  static const uint32_t Function          = 7;
  static const uint32_t Generic           = 8;
  static const uint32_t PushConstant      = 9;
  static const uint32_t AtomicCounter     = 10;
  static const uint32_t Image             = 11;
  static const uint32_t StorageBuffer     = 12;
  static const uint32_t PhysicalStorageBuffer = 5349;
}

// Capabilities
namespace Cap {
  static const uint32_t Matrix                        = 0;
  static const uint32_t Shader                        = 1;
  static const uint32_t Addresses                     = 4;
  static const uint32_t Linkage                       = 5;
  static const uint32_t Kernel                        = 6;
  static const uint32_t Int64                         = 11;
  static const uint32_t Int8                          = 39;
  static const uint32_t GenericPointer                = 38;
  static const uint32_t VariablePointers              = 4442;
  static const uint32_t PhysicalStorageBufferAddresses = 5347;
  static const uint32_t Int16                         = 22;
  static const uint32_t Float16                       = 9;
  static const uint32_t Float64                       = 10;
  static const uint32_t StorageBuffer8BitAccess       = 4448;
  static const uint32_t UniformAndStorageBuffer8BitAccess = 4449;
  static const uint32_t StorageBuffer16BitAccess      = 4433;
}

// Decorations
namespace Deco {
  static const uint32_t SpecId          = 1;
  static const uint32_t Block           = 2;
  static const uint32_t BuiltIn         = 11;
  static const uint32_t Binding         = 33;
  static const uint32_t DescriptorSet   = 34;
  static const uint32_t Offset          = 35;
  static const uint32_t FuncParamAttr   = 38;
  static const uint32_t Alignment       = 44;
  static const uint32_t ArrayStride     = 6;
}

// BuiltIns
namespace Builtin {
  static const uint32_t WorkgroupSize   = 25;
}

// FuncParamAttr
namespace FPA {
  static const uint32_t ByVal = 2;
}

// NonSemantic.ClspvReflection instruction IDs
namespace NSRefl {
  static const uint32_t Kernel                        = 1;
  static const uint32_t ArgumentInfo                  = 2;
  static const uint32_t ArgumentStorageBuffer         = 3;
  static const uint32_t ArgumentPodPushConstant       = 7;
  static const uint32_t SpecConstantWorkgroupSize     = 12;
  static const uint32_t ArgumentPointerPushConstant   = 26;
  static const uint32_t ProgramScopeVariablesStorageBuffer = 28;
}

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------
using Words = std::vector<uint32_t>;

struct Instr {
  uint32_t opcode = 0;
  Words    words; // all words including opcode+wc word
  // Convenience accessors
  uint32_t wc()  const { return words[0] >> 16; }
  uint32_t word(size_t i) const { return words[i]; }
};

// Parse a SPIR-V instruction from words[pos], return next pos
static size_t parseInstr(const Words& spv, size_t pos, Instr& out) {
  uint32_t w0 = spv[pos];
  uint32_t wc = w0 >> 16;
  out.opcode = w0 & 0xffff;
  out.words.assign(spv.begin() + pos, spv.begin() + pos + wc);
  return pos + wc;
}

// Encode a SPIR-V literal string (null-terminated, padded to 4 bytes)
static Words encodeLiteralString(const std::string& s) {
  Words result;
  uint32_t w = 0;
  int shift = 0;
  for (char c : s) {
    w |= (uint32_t)(unsigned char)c << shift;
    shift += 8;
    if (shift == 32) {
      result.push_back(w);
      w = 0;
      shift = 0;
    }
  }
  // null terminator + padding
  result.push_back(w); // includes the null byte (shift < 32 means null already in there)
  return result;
}

// Decode a SPIR-V literal string from words starting at offset
static std::string decodeLiteralString(const Words& words, size_t offset) {
  std::string result;
  for (size_t i = offset; i < words.size(); ++i) {
    uint32_t w = words[i];
    for (int shift = 0; shift < 32; shift += 8) {
      char c = (char)((w >> shift) & 0xff);
      if (c == 0) return result;
      result += c;
    }
  }
  return result;
}

// Build instruction word array: [opcode|(wc<<16), operand0, operand1, ...]
static Words makeInstr(uint16_t opcode, std::initializer_list<uint32_t> operands) {
  Words w;
  w.push_back(0); // placeholder
  for (uint32_t op : operands) w.push_back(op);
  w[0] = ((uint32_t)w.size() << 16) | opcode;
  return w;
}

static Words makeInstr(uint16_t opcode, const Words& operands) {
  Words w;
  w.push_back(0);
  for (uint32_t op : operands) w.push_back(op);
  w[0] = ((uint32_t)w.size() << 16) | opcode;
  return w;
}

// Append words of an instruction to output
static void emit(Words& out, const Words& instr) {
  out.insert(out.end(), instr.begin(), instr.end());
}

static void emitInstr(Words& out, uint16_t opcode, std::initializer_list<uint32_t> operands) {
  emit(out, makeInstr(opcode, operands));
}

static void emitInstr(Words& out, uint16_t opcode, const Words& operands) {
  emit(out, makeInstr(opcode, operands));
}

// ---------------------------------------------------------------------------
// ModuleInfo — collected during Pass 1
// ---------------------------------------------------------------------------

struct PtrTypeInfo {
  uint32_t storage_class = 0;
  uint32_t base_type     = 0;
};

struct ConstantInfo {
  uint32_t type_id = 0;
  uint64_t value   = 0;
};

struct ParamInfo {
  uint32_t id        = 0;
  uint32_t type_id   = 0;
  bool     is_byval  = false; // has FuncParamAttr ByVal decoration
  uint32_t byval_type_id = 0; // if byval, the struct type (base of pointer)
  // Classification after analysis:
  bool     is_crossworkgroup_ptr = false; // param type is ptr to CrossWorkgroup
  bool     is_generic_ptr        = false; // param type is ptr to Generic
  bool     is_pointer_kind       = false; // used as ptr in loads/stores (not just ConvertPtrToU)
  bool     is_pod_kind           = false; // only used via ConvertPtrToU → becomes ulong
};

struct FunctionInfo {
  uint32_t id         = 0;
  uint32_t ret_type   = 0;
  uint32_t fn_type    = 0;
  bool     is_kernel_stub = false; // single-block, single FunctionCall
  uint32_t impl_call_target = 0;   // if stub: the impl fn called
  std::vector<ParamInfo> params;
  std::vector<Instr> body; // all instructions in function body (after params)
};

struct EntryPointInfo {
  uint32_t fn_id = 0;
  std::string name;
  std::vector<uint32_t> interface_ids;
};

struct ChipVarInfo {
  uint32_t id          = 0;
  std::string name;
  uint32_t ptr_type_id = 0; // the OpTypePointer CrossWorkgroup id
  uint32_t base_type_id = 0; // the base type
  // Assigned in Pass 2:
  uint32_t ssbo_struct_type_id = 0;
  uint32_t ssbo_ptr_type_id    = 0;
  uint32_t ssbo_var_id         = 0;
  uint32_t descriptor_set      = 0;
  uint32_t binding              = 0;
};

struct ModuleInfo {
  uint32_t version = 0;
  uint32_t bound   = 0;
  uint32_t generator = 0;
  uint32_t schema  = 0;

  // Types
  std::unordered_map<uint32_t, PtrTypeInfo>         ptr_types;    // id -> {sc, base}
  std::unordered_map<uint32_t, std::vector<uint32_t>> struct_members; // struct id -> member type ids
  std::unordered_map<uint32_t, uint32_t>            int_widths;   // id -> width
  std::unordered_map<uint32_t, uint32_t>            float_widths;
  std::unordered_map<uint32_t, std::pair<uint32_t,uint32_t>> vec_types; // id -> {elem,count}
  std::unordered_map<uint32_t, uint32_t>            arr_types;    // id -> elem type
  std::unordered_map<uint32_t, uint32_t>            fn_types;     // id -> ret type (simplified)
  std::unordered_map<uint32_t, std::vector<uint32_t>> fn_type_params; // id -> param type ids
  uint32_t void_type_id  = 0;

  // ExtInstImport IDs from the input module (e.g. OpenCL.std, GLSL.std.450)
  std::unordered_set<uint32_t> ext_inst_import_ids;

  // Names and decorations
  std::unordered_map<uint32_t, std::string> names;
  std::unordered_map<uint32_t, std::vector<std::vector<uint32_t>>> decorations; // id -> list of deco words (without opcode/wc)
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::vector<std::vector<uint32_t>>>> member_decorations; // struct_id -> {member_idx -> deco words}
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>> member_offsets; // struct_id -> {member_idx -> offset}

  // Constants
  std::unordered_map<uint32_t, ConstantInfo> constants; // id -> {type_id, value}

  // Entry points
  std::vector<EntryPointInfo> entry_points;

  // Functions (all)
  std::unordered_map<uint32_t, FunctionInfo> functions;
  std::vector<uint32_t> function_order; // order they appear

  // Global variables
  std::vector<ChipVarInfo> chip_vars; // __chip_var_* and __chip_module_has_no_IGBAs
  std::unordered_map<uint32_t, size_t> chip_var_by_id; // id -> index in chip_vars

  // UniformConstant variables named __chip_var_* that hold initializer constants.
  // These are used in __chip_var_init_* functions as CopyMemorySized sources.
  // Maps: variable id -> initializer constant id
  std::unordered_map<uint32_t, uint32_t> init_var_constants;

  // Set of IDs for kernel stub functions and impl functions
  std::unordered_set<uint32_t> stub_fn_ids;
  std::unordered_set<uint32_t> impl_fn_ids;

  // Capabilities present in input
  std::unordered_set<uint32_t> capabilities;

  // Extension names declared in input (from OpExtension)
  std::vector<std::string> extensions;

  // All original instructions by section (for output)
  std::vector<Instr> debug_instrs;   // OpName, OpMemberName, OpString
  std::vector<Instr> source_instrs;  // OpSource
  std::vector<Instr> original_types; // all type/const/var instructions

  // ID for uint32 and uint64 types (found or created)
  uint32_t uint32_type_id = 0;
  uint32_t uint64_type_id = 0;
  uint32_t uint8_type_id  = 0;

  // Set when the module has CrossWorkgroup globals that are not chip_vars.
  // These cannot be transformed to PSB Vulkan SPIR-V; the caller should fall
  // back to clvk's internal clspv pipeline.
  bool has_unsupported_globals = false;

  // BuiltIn Input variables: var_id → BuiltIn kind (e.g. 26=WorkgroupId, 27=LocalInvocationId)
  std::unordered_map<uint32_t, uint32_t> builtin_input_vars;
  // Original pointer type ID for each BuiltIn Input variable
  std::unordered_map<uint32_t, uint32_t> builtin_input_ptr_types; // var_id → ptr_type_id
  // WorkgroupSize variable ID — to be removed (replaced by SpecConstantComposite)
  uint32_t workgroup_size_var_id = 0;
};

// ---------------------------------------------------------------------------
// Pass 1: scan SPIR-V and collect ModuleInfo
// ---------------------------------------------------------------------------

static bool collectModuleInfo(const Words& spv, ModuleInfo& info, std::string& err) {
  if (spv.size() < 5) { err = "SPIR-V too short"; return false; }
  if (spv[0] != spv::MagicNumber) { err = "Bad magic number"; return false; }

  info.version   = spv[1];
  info.generator = spv[2];
  info.bound     = spv[3];
  info.schema    = spv[4];

  // First pass: collect all type/name/decoration/constant info
  // and identify functions
  size_t pos = 5;
  FunctionInfo* cur_fn = nullptr;
  bool in_function = false;

  while (pos < spv.size()) {
    Instr instr;
    pos = parseInstr(spv, pos, instr);

    switch (instr.opcode) {
    case Op::Capability:
      if (instr.wc() >= 2)
        info.capabilities.insert(instr.word(1));
      break;

    case Op::Extension:
      if (instr.wc() >= 2)
        info.extensions.push_back(decodeLiteralString(instr.words, 1));
      break;

    case Op::ExtInstImport:
      if (instr.wc() >= 2)
        info.ext_inst_import_ids.insert(instr.word(1));
      break;

    case Op::Source:
      info.source_instrs.push_back(instr);
      break;

    case Op::Name:
      if (instr.wc() >= 3) {
        uint32_t id = instr.word(1);
        info.names[id] = decodeLiteralString(instr.words, 2);
        info.debug_instrs.push_back(instr);
      }
      break;

    case Op::MemberName:
      if (instr.wc() >= 4)
        info.debug_instrs.push_back(instr);
      break;

    case Op::String:
      info.debug_instrs.push_back(instr);
      break;

    case Op::Decorate:
      if (instr.wc() >= 3) {
        uint32_t id = instr.word(1);
        std::vector<uint32_t> deco(instr.words.begin() + 2, instr.words.end());
        info.decorations[id].push_back(deco);
      }
      break;

    case Op::MemberDecorate:
      if (instr.wc() >= 4) {
        uint32_t struct_id  = instr.word(1);
        uint32_t member_idx = instr.word(2);
        uint32_t deco_kind  = instr.word(3);
        std::vector<uint32_t> deco(instr.words.begin() + 3, instr.words.end());
        info.member_decorations[struct_id][member_idx].push_back(deco);
        if (deco_kind == Deco::Offset && instr.wc() >= 5) {
          info.member_offsets[struct_id][member_idx] = instr.word(4);
        }
      }
      break;

    case Op::TypeVoid:
      if (instr.wc() >= 2) {
        info.void_type_id = instr.word(1);
        info.original_types.push_back(instr);
      }
      break;

    case Op::TypeInt:
      if (instr.wc() >= 4) {
        uint32_t id    = instr.word(1);
        uint32_t width = instr.word(2);
        // uint32_t sign  = instr.word(3);
        info.int_widths[id] = width;
        if (width == 32) info.uint32_type_id = id;
        if (width == 64) info.uint64_type_id = id;
        if (width == 8)  info.uint8_type_id  = id;
        info.original_types.push_back(instr);
      }
      break;

    case Op::TypeFloat:
      if (instr.wc() >= 3) {
        uint32_t id    = instr.word(1);
        uint32_t width = instr.word(2);
        info.float_widths[id] = width;
        info.original_types.push_back(instr);
      }
      break;

    case Op::TypeVector:
      if (instr.wc() >= 4) {
        uint32_t id   = instr.word(1);
        uint32_t elem = instr.word(2);
        uint32_t cnt  = instr.word(3);
        info.vec_types[id] = {elem, cnt};
        info.original_types.push_back(instr);
      }
      break;

    case Op::TypeArray:
      if (instr.wc() >= 4) {
        uint32_t id   = instr.word(1);
        uint32_t elem = instr.word(2);
        info.arr_types[id] = elem;
        info.original_types.push_back(instr);
      }
      break;

    case Op::TypeStruct:
      if (instr.wc() >= 2) {
        uint32_t id = instr.word(1);
        std::vector<uint32_t> members;
        for (uint32_t i = 2; i < instr.wc(); ++i)
          members.push_back(instr.word(i));
        info.struct_members[id] = members;
        info.original_types.push_back(instr);
      }
      break;

    case Op::TypePointer:
      if (instr.wc() >= 4) {
        uint32_t id   = instr.word(1);
        uint32_t sc   = instr.word(2);
        uint32_t base = instr.word(3);
        info.ptr_types[id] = {sc, base};
        info.original_types.push_back(instr);
      }
      break;

    case Op::TypeFunction:
      if (instr.wc() >= 3) {
        uint32_t id  = instr.word(1);
        uint32_t ret = instr.word(2);
        info.fn_types[id] = ret;
        std::vector<uint32_t> params;
        for (uint32_t i = 3; i < instr.wc(); ++i)
          params.push_back(instr.word(i));
        info.fn_type_params[id] = params;
        info.original_types.push_back(instr);
      }
      break;

    case Op::Constant:
      if (instr.wc() >= 4) {
        uint32_t id = instr.word(2);
        uint64_t val = instr.word(3);
        if (instr.wc() >= 5) val |= ((uint64_t)instr.word(4) << 32);
        info.constants[id] = {instr.word(1), val};
        info.original_types.push_back(instr);
      }
      break;

    case 1:  // OpUndef — can appear at global scope or inside functions
    case 41: // OpConstantTrue
    case 42: // OpConstantFalse
    case 46: // OpConstantNull
    case Op::SpecConstant:
    case Op::ConstantComposite:
    case Op::SpecConstantComposite:
      if (!in_function && instr.wc() >= 2)
        info.original_types.push_back(instr);
      else if (in_function && cur_fn)
        cur_fn->body.push_back(instr);
      break;

    // Other type declarations not explicitly handled above (e.g. TypeBool, TypeRuntimeArray, etc.)
    // Must be added to original_types so they're emitted in the output.
    case 20: // OpTypeBool
    case 25: // OpTypeRuntimeArray
    case 26: // OpTypeImage
    case 27: // OpTypeSampler
    case 29: // OpTypeSampledImage
      if (!in_function)
        info.original_types.push_back(instr);
      break;

    case Op::Variable:
      if (instr.wc() >= 4) {
        uint32_t ptr_type = instr.word(1);
        uint32_t id       = instr.word(2);
        uint32_t sc       = instr.word(3);
        if (sc == SC::CrossWorkgroup) {
          auto it = info.names.find(id);
          if (it != info.names.end()) {
            const std::string& n = it->second;
            if (n.substr(0, 11) == "__chip_var_" || n == "__chip_module_has_no_IGBAs") {
              ChipVarInfo cv;
              cv.id           = id;
              cv.name         = n;
              cv.ptr_type_id  = ptr_type;
              auto pit = info.ptr_types.find(ptr_type);
              if (pit != info.ptr_types.end())
                cv.base_type_id = pit->second.base_type;
              info.chip_var_by_id[id] = info.chip_vars.size();
              info.chip_vars.push_back(cv);
              // Don't add to original_types — we'll replace with SSBO
              break;
            }
          }
          // CrossWorkgroup global that is not a recognized chip_var
          // (e.g. _ZL18__chip_clk_counter, __chipspv_device_heap).
          // We cannot transform these to PSB Vulkan SPIR-V; signal
          // that the caller should fall back to clvk's clspv pipeline.
          info.has_unsupported_globals = true;
          break;
        }
        // Track UniformConstant __chip_var_* variables: these hold constant
        // initializer data used by __chip_var_init_* functions via CopyMemorySized.
        // Do NOT emit them in the Vulkan output (UniformConstant can only be sampler/image).
        if (sc == SC::UniformConstant && instr.wc() >= 5) {
          auto it = info.names.find(id);
          if (it != info.names.end() && it->second.substr(0, 11) == "__chip_var_") {
            info.init_var_constants[id] = instr.word(4); // initializer constant ID
            break; // skip adding to original_types
          }
        }
        info.original_types.push_back(instr);
      }
      break;

    case Op::EntryPoint:
      if (instr.wc() >= 4) {
        EntryPointInfo ep;
        // word(1) = execution model, word(2) = fn_id, word(3+) = name string then interface vars
        ep.fn_id = instr.word(2);
        size_t str_start = 3;
        // decode name
        ep.name = decodeLiteralString(instr.words, str_start);
        // compute how many words the name takes
        size_t name_words = (ep.name.size() + 4) / 4; // ceil to 4
        size_t iface_start = str_start + name_words;
        for (size_t i = iface_start; i < instr.wc(); ++i)
          ep.interface_ids.push_back(instr.word(i));
        info.entry_points.push_back(ep);
      }
      break;

    case Op::Function:
      if (instr.wc() >= 4) {
        uint32_t id      = instr.word(2);
        uint32_t ret     = instr.word(1);
        uint32_t fn_type = instr.word(4);
        FunctionInfo fn;
        fn.id      = id;
        fn.ret_type = ret;
        fn.fn_type  = fn_type;
        info.functions[id] = fn;
        info.function_order.push_back(id);
        cur_fn = &info.functions[id];
        in_function = true;
      }
      break;

    case Op::FunctionParameter:
      if (in_function && cur_fn && instr.wc() >= 3) {
        ParamInfo p;
        p.type_id = instr.word(1);
        p.id      = instr.word(2);
        cur_fn->params.push_back(p);
      }
      break;

    case Op::FunctionEnd:
      if (in_function && cur_fn) {
        cur_fn->body.push_back(instr);
        in_function = false;
        cur_fn = nullptr;
      }
      break;

    default:
      if (in_function && cur_fn) {
        cur_fn->body.push_back(instr);
      }
      break;
    }
  }

  // Determine parameter attributes (ByVal) from decorations
  for (auto& [fn_id, fn] : info.functions) {
    for (auto& param : fn.params) {
      auto dit = info.decorations.find(param.id);
      if (dit != info.decorations.end()) {
        for (auto& deco : dit->second) {
          if (!deco.empty() && deco[0] == Deco::FuncParamAttr && deco.size() >= 2 && deco[1] == FPA::ByVal) {
            param.is_byval = true;
            // The param type is pointer to the struct
            auto pit = info.ptr_types.find(param.type_id);
            if (pit != info.ptr_types.end()) {
              param.byval_type_id = pit->second.base_type;
            }
          }
        }
      }
      // Classify CrossWorkgroup pointer params
      auto pit = info.ptr_types.find(param.type_id);
      if (pit != info.ptr_types.end()) {
        if (pit->second.storage_class == SC::CrossWorkgroup)
          param.is_crossworkgroup_ptr = true;
        if (pit->second.storage_class == SC::Generic)
          param.is_generic_ptr = true;
      }
    }
  }

  // Identify kernel stubs vs impl functions
  // A stub has entry point listed + body with exactly one FunctionCall
  std::unordered_set<uint32_t> ep_fn_ids;
  for (auto& ep : info.entry_points)
    ep_fn_ids.insert(ep.fn_id);

  for (auto& [fn_id, fn] : info.functions) {
    if (ep_fn_ids.count(fn_id) == 0) continue;
    // Check if body has exactly one FunctionCall
    int call_count = 0;
    uint32_t call_target = 0;
    for (auto& instr : fn.body) {
      if (instr.opcode == Op::FunctionCall && instr.wc() >= 4) {
        call_count++;
        call_target = instr.word(3);
      }
    }
    if (call_count == 1) {
      fn.is_kernel_stub = true;
      fn.impl_call_target = call_target;
      info.stub_fn_ids.insert(fn_id);
      info.impl_fn_ids.insert(call_target);
    }
  }

  // For impl functions: classify each CrossWorkgroup param as Pod or Pointer
  // by scanning the function body for uses
  for (auto& fn_id : info.impl_fn_ids) {
    auto it = info.functions.find(fn_id);
    if (it == info.functions.end()) continue;
    FunctionInfo& fn = it->second;

    // Build set of CrossWorkgroup param ids
    std::unordered_set<uint32_t> cw_params;
    for (auto& p : fn.params) {
      if (p.is_crossworkgroup_ptr) cw_params.insert(p.id);
    }
    if (cw_params.empty()) continue;

    // Track which params are used in ConvertPtrToU (pod) vs as pointer base (pointer kind)
    std::unordered_set<uint32_t> pod_params;
    std::unordered_set<uint32_t> ptr_params;

    for (auto& instr : fn.body) {
      if (instr.opcode == Op::ConvertPtrToU && instr.wc() >= 4) {
        uint32_t src = instr.word(3);
        if (cw_params.count(src)) pod_params.insert(src);
      }
      // Check if param used directly as pointer in load/store/access chain
      if ((instr.opcode == Op::Load || instr.opcode == Op::Store ||
           instr.opcode == Op::AccessChain || instr.opcode == Op::InBoundsAccessChain ||
           instr.opcode == Op::PtrAccessChain || instr.opcode == Op::InBoundsPtrAccessChain) &&
          instr.wc() >= 3) {
        // Base pointer is typically word(3) for load/store and word(3) for access chains
        uint32_t base_idx = (instr.opcode == Op::Store) ? 1 : 3;
        if (base_idx < instr.wc()) {
          uint32_t base = instr.word(base_idx);
          if (cw_params.count(base)) ptr_params.insert(base);
        }
      }
      // CW ptr passed as argument to a FunctionCall is also a pointer use:
      // the callee receives it as a CW/PSB ptr, not a raw integer address.
      if (instr.opcode == Op::FunctionCall && instr.wc() >= 4) {
        for (uint32_t ai = 3; ai < instr.wc(); ++ai) {
          if (cw_params.count(instr.word(ai)))
            ptr_params.insert(instr.word(ai));
        }
      }
    }

    // If a param is only in pod_params (not ptr_params), it's pod kind
    // Otherwise pointer kind
    for (auto& p : fn.params) {
      if (!p.is_crossworkgroup_ptr) continue;
      bool pod_use = pod_params.count(p.id) > 0;
      bool ptr_use = ptr_params.count(p.id) > 0;
      if (!pod_use && !ptr_use) {
        // No use found — treat as pod (conservative)
        p.is_pod_kind = true;
      } else if (ptr_use) {
        p.is_pointer_kind = true;
      } else {
        p.is_pod_kind = true;
      }
    }
  }

  // Identify BuiltIn Input variables from collected decorations.
  // These need special handling: retype v3ulong → v3uint, emit BuiltIn decorations.
  for (auto& instr : info.original_types) {
    if (instr.opcode != Op::Variable || instr.wc() < 4) continue;
    uint32_t ptr_type_id = instr.word(1);
    uint32_t var_id = instr.word(2);
    uint32_t sc     = instr.word(3);
    if (sc != SC::Input) continue;
    auto dit = info.decorations.find(var_id);
    if (dit == info.decorations.end()) continue;
    for (auto& deco : dit->second) {
      if (deco.size() >= 2 && deco[0] == Deco::BuiltIn) {
        uint32_t builtin_kind = deco[1];
        info.builtin_input_vars[var_id] = builtin_kind;
        info.builtin_input_ptr_types[var_id] = ptr_type_id;
        if (builtin_kind == 25 /*WorkgroupSize*/)
          info.workgroup_size_var_id = var_id;
        break;
      }
    }
  }

  return true;
}

// ---------------------------------------------------------------------------
// Type size/alignment (std430 rules for push constants)
// ---------------------------------------------------------------------------

static uint32_t typeSize(uint32_t type_id, const ModuleInfo& info);
static uint32_t typeAlign(uint32_t type_id, const ModuleInfo& info);

static uint32_t typeSize(uint32_t type_id, const ModuleInfo& info) {
  if (type_id == info.void_type_id) return 0;
  auto iit = info.int_widths.find(type_id);
  if (iit != info.int_widths.end()) return iit->second / 8;
  auto fit = info.float_widths.find(type_id);
  if (fit != info.float_widths.end()) return fit->second / 8;
  auto vit = info.vec_types.find(type_id);
  if (vit != info.vec_types.end()) {
    uint32_t elem_size = typeSize(vit->second.first, info);
    uint32_t cnt = vit->second.second;
    // vec3 is padded to vec4 size
    if (cnt == 3) cnt = 4;
    return elem_size * cnt;
  }
  auto sit = info.struct_members.find(type_id);
  if (sit != info.struct_members.end()) {
    uint32_t offset = 0;
    uint32_t align  = typeAlign(type_id, info);
    for (uint32_t m : sit->second) {
      uint32_t ma = typeAlign(m, info);
      offset = (offset + ma - 1) & ~(ma - 1);
      offset += typeSize(m, info);
    }
    // Pad to struct alignment
    offset = (offset + align - 1) & ~(align - 1);
    return offset;
  }
  auto pit = info.ptr_types.find(type_id);
  if (pit != info.ptr_types.end()) {
    // PSB pointers are 64-bit
    return 8;
  }
  // Unknown type — return 4 as fallback
  return 4;
}

static uint32_t typeAlign(uint32_t type_id, const ModuleInfo& info) {
  if (type_id == info.void_type_id) return 1;
  auto iit = info.int_widths.find(type_id);
  if (iit != info.int_widths.end()) return iit->second / 8;
  auto fit = info.float_widths.find(type_id);
  if (fit != info.float_widths.end()) return fit->second / 8;
  auto vit = info.vec_types.find(type_id);
  if (vit != info.vec_types.end()) {
    uint32_t elem_align = typeAlign(vit->second.first, info);
    uint32_t cnt = vit->second.second;
    // vec3 alignment = vec4 alignment
    if (cnt == 3) cnt = 4;
    return elem_align * cnt;
  }
  auto sit = info.struct_members.find(type_id);
  if (sit != info.struct_members.end()) {
    uint32_t max_align = 4; // minimum
    for (uint32_t m : sit->second) {
      uint32_t ma = typeAlign(m, info);
      if (ma > max_align) max_align = ma;
    }
    return max_align;
  }
  auto pit = info.ptr_types.find(type_id);
  if (pit != info.ptr_types.end()) return 8;
  return 4;
}

// Compute the push constant struct layout for a kernel's arguments
struct ArgLayout {
  uint32_t type_id     = 0;   // original type (ulong for CW pointer, struct for byval)
  uint32_t pc_offset   = 0;
  uint32_t pc_size     = 0;
  bool     is_byval    = false;
  bool     is_ulong    = false; // was a CW pointer, now a ulong in PC
};

static std::vector<ArgLayout> computeKernelPCLayout(const FunctionInfo& impl_fn,
                                                     const ModuleInfo& info) {
  std::vector<ArgLayout> layout;
  uint32_t offset = 0;

  for (auto& param : impl_fn.params) {
    ArgLayout al;
    uint32_t type_id;
    uint32_t sz, align;

    if (param.is_crossworkgroup_ptr) {
      // Becomes ulong in push constant
      type_id  = info.uint64_type_id;
      sz       = 8;
      align    = 8;
      al.is_ulong = true;
    } else if (param.is_byval) {
      type_id  = param.byval_type_id;
      sz       = typeSize(type_id, info);
      align    = typeAlign(type_id, info);
      al.is_byval = true;
    } else {
      type_id  = param.type_id;
      sz       = typeSize(type_id, info);
      align    = typeAlign(type_id, info);
    }
    if (align < 1) align = 1;
    offset = (offset + align - 1) & ~(align - 1);

    al.type_id   = type_id;
    al.pc_offset = offset;
    al.pc_size   = sz;
    layout.push_back(al);

    offset += sz;
  }
  return layout;
}

// ---------------------------------------------------------------------------
// Check if a struct type (recursively) contains pointer members to
// CrossWorkgroup or Generic storage class
// ---------------------------------------------------------------------------
static bool structHasBadPointers(uint32_t type_id, const ModuleInfo& info, int depth = 0) {
  if (depth > 16) return false;
  auto sit = info.struct_members.find(type_id);
  if (sit == info.struct_members.end()) return false;
  for (uint32_t m : sit->second) {
    auto pit = info.ptr_types.find(m);
    if (pit != info.ptr_types.end()) {
      uint32_t sc = pit->second.storage_class;
      if (sc == SC::CrossWorkgroup || sc == SC::Generic) return true;
    }
    if (structHasBadPointers(m, info, depth + 1)) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// CFG structurizer: add OpSelectionMerge / OpLoopMerge to function bodies
// that lack them (OpenCL SPIR-V from llvm-spirv).
// ---------------------------------------------------------------------------

// Returns true if 'opcode' is a block terminator.
static bool isTerminator(uint32_t op) {
  return op == Op::Branch || op == Op::BranchConditional ||
         op == Op::Switch || op == Op::Return ||
         op == Op::ReturnValue || op == Op::Unreachable;
}

// Add structured-CF markers to function body instructions.
// 'body' contains all instructions from (and including) the first OpLabel to
// (and including) the OpFunctionEnd.  Returns the modified instruction list.
// 'next_id' is used to allocate new IDs for synthetic unreachable merge blocks.
static std::vector<Instr> structurizeFunctionBody(const std::vector<Instr>& body,
                                                   uint32_t* next_id = nullptr) {
  // Parse into basic blocks.
  struct Block {
    uint32_t label_id = 0;
    std::vector<size_t> instr_idx; // indices into 'body'
    std::vector<uint32_t> succs;   // successor label IDs
  };

  std::vector<Block> blocks;
  std::unordered_map<uint32_t, size_t> label_to_blk; // label_id → block index

  for (size_t i = 0; i < body.size(); ++i) {
    const Instr& ins = body[i];
    if (ins.opcode == Op::FunctionEnd) break;
    if (ins.opcode == Op::Label && ins.wc() >= 2) {
      blocks.push_back({});
      Block& b = blocks.back();
      b.label_id = ins.word(1);
      label_to_blk[b.label_id] = blocks.size() - 1;
      b.instr_idx.push_back(i);
    } else if (!blocks.empty()) {
      blocks.back().instr_idx.push_back(i);
    }
  }
  if (blocks.empty()) return body;

  // Build successor lists from terminators.
  for (auto& blk : blocks) {
    if (blk.instr_idx.empty()) continue;
    const Instr& term = body[blk.instr_idx.back()];
    if (term.opcode == Op::Branch && term.wc() >= 2)
      blk.succs = {term.word(1)};
    else if (term.opcode == Op::BranchConditional && term.wc() >= 4) {
      uint32_t t1 = term.word(2), t2 = term.word(3);
      blk.succs = {t1};
      if (t2 != t1) blk.succs.push_back(t2);
    } else if (term.opcode == Op::Switch && term.wc() >= 3) {
      blk.succs = {term.word(2)};
      for (uint32_t k = 3; k + 1 < term.wc(); k += 2)
        blk.succs.push_back(term.word(k + 1));
      // Deduplicate
      std::sort(blk.succs.begin(), blk.succs.end());
      blk.succs.erase(std::unique(blk.succs.begin(), blk.succs.end()), blk.succs.end());
    }
  }

  // Iterative DFS to compute post-order (→ RPO = reverse of post-order).
  // Uses explicit stack to avoid system stack overflow on large functions.
  std::vector<int> rpo_num(blocks.size(), -1); // block index → RPO number
  {
    std::vector<bool> visited(blocks.size(), false);
    std::vector<int> post;
    // Stack entry: (block_index, next_successor_to_process)
    auto iterDFS = [&](size_t start) {
      struct Frame { size_t bi; size_t succ_idx; };
      std::vector<Frame> stk;
      if (visited[start]) return;
      stk.push_back({start, 0});
      visited[start] = true;
      while (!stk.empty()) {
        Frame& fr = stk.back();
        bool pushed = false;
        while (fr.succ_idx < blocks[fr.bi].succs.size()) {
          uint32_t s = blocks[fr.bi].succs[fr.succ_idx++];
          auto it = label_to_blk.find(s);
          if (it == label_to_blk.end()) continue;
          size_t sbi = it->second;
          if (!visited[sbi]) {
            visited[sbi] = true;
            stk.push_back({sbi, 0});
            pushed = true;
            break;
          }
        }
        if (!pushed) {
          post.push_back((int)fr.bi);
          stk.pop_back();
        }
      }
    };
    if (!blocks.empty()) iterDFS(0);
    for (size_t i = 0; i < blocks.size(); ++i) iterDFS(i);
    int rpo = 0;
    for (int i = (int)post.size() - 1; i >= 0; --i)
      rpo_num[post[i]] = rpo++;
  }

  // Detect loop headers: a block B is a loop header if some successor S → B
  // has rpo_num[S] >= rpo_num[B] (i.e., S comes later in RPO = back-edge source).
  std::unordered_set<uint32_t> loop_header_labels;
  // back_edge_source[loop_header_label] = continue_label
  std::unordered_map<uint32_t, uint32_t> loop_continue;
  for (auto& blk : blocks) {
    auto bi_it = label_to_blk.find(blk.label_id);
    if (bi_it == label_to_blk.end()) continue;
    int src_rpo = rpo_num[bi_it->second];
    for (uint32_t s : blk.succs) {
      auto ti = label_to_blk.find(s);
      if (ti == label_to_blk.end()) continue;
      int tgt_rpo = rpo_num[ti->second];
      if (tgt_rpo >= 0 && src_rpo >= 0 && tgt_rpo <= src_rpo) {
        // back-edge: blk → s (s is loop header)
        loop_header_labels.insert(s);
        loop_continue[s] = blk.label_id;
      }
    }
  }

  // Forward-reachability helper: find first label reachable from both t1 and t2
  // (explores in RPO order to prefer the nearest post-dominator).
  auto findMerge = [&](uint32_t t1, uint32_t t2,
                        const std::unordered_set<uint32_t>& stop_at) -> uint32_t {
    if (t1 == t2) return t1;
    // BFS from t1
    std::unordered_set<uint32_t> from_t1;
    {
      std::queue<uint32_t> q;
      q.push(t1);
      while (!q.empty()) {
        uint32_t cur = q.front(); q.pop();
        if (!from_t1.insert(cur).second) continue;
        if (stop_at.count(cur)) continue;
        auto it = label_to_blk.find(cur);
        if (it == label_to_blk.end()) continue;
        for (uint32_t s : blocks[it->second].succs) q.push(s);
      }
    }
    // BFS from t2, return first block in from_t1 (in RPO order → pick min RPO)
    uint32_t best = 0;
    int best_rpo = INT_MAX;
    std::queue<uint32_t> q2;
    q2.push(t2);
    std::unordered_set<uint32_t> vis;
    while (!q2.empty()) {
      uint32_t cur = q2.front(); q2.pop();
      if (!vis.insert(cur).second) continue;
      if (from_t1.count(cur)) {
        auto it = label_to_blk.find(cur);
        int r = (it != label_to_blk.end()) ? rpo_num[it->second] : INT_MAX;
        if (r < best_rpo) { best_rpo = r; best = cur; }
      }
      if (stop_at.count(cur)) continue;
      auto it = label_to_blk.find(cur);
      if (it == label_to_blk.end()) continue;
      for (uint32_t s : blocks[it->second].succs) q2.push(s);
    }
    return best;
  };

  // Compute loop merge blocks.
  // For a loop header H with BranchConditional(cond, body, exit):
  //   the exit target is the first successor NOT inside the loop body.
  std::unordered_map<uint32_t, uint32_t> loop_merge_label; // header → merge
  for (uint32_t header : loop_header_labels) {
    auto hi = label_to_blk.find(header);
    if (hi == label_to_blk.end()) continue;
    Block& hblk = blocks[hi->second];
    int h_rpo = rpo_num[hi->second];
    uint32_t cont = loop_continue.count(header) ? loop_continue[header] : 0;
    int cont_rpo = 0;
    if (cont) {
      auto ci = label_to_blk.find(cont);
      cont_rpo = (ci != label_to_blk.end()) ? rpo_num[ci->second] : 0;
    }
    const Instr& term = body[hblk.instr_idx.back()];
    if (term.opcode == Op::BranchConditional && term.wc() >= 4) {
      uint32_t t1 = term.word(2), t2 = term.word(3);
      // The merge is the target that is OUTSIDE the loop (RPO > cont or RPO < h)
      auto t1i = label_to_blk.find(t1);
      auto t2i = label_to_blk.find(t2);
      int r1 = (t1i != label_to_blk.end()) ? rpo_num[t1i->second] : -1;
      int r2 = (t2i != label_to_blk.end()) ? rpo_num[t2i->second] : -1;
      // The body target should have r >= h_rpo and r <= cont_rpo
      // The merge target should be outside that range
      if (r1 >= 0 && r1 > cont_rpo)
        loop_merge_label[header] = t1;
      else if (r2 >= 0 && r2 > cont_rpo)
        loop_merge_label[header] = t2;
      else if (r1 < h_rpo)
        loop_merge_label[header] = t1;
      else if (r2 < h_rpo)
        loop_merge_label[header] = t2;
      else
        loop_merge_label[header] = (r1 > r2) ? t1 : t2; // fallback
    } else if (term.opcode == Op::Branch && term.wc() >= 2) {
      // Unconditional header (rare) — merge is target if it's outside the loop
      uint32_t tgt = term.word(1);
      auto ti = label_to_blk.find(tgt);
      int rt = (ti != label_to_blk.end()) ? rpo_num[ti->second] : -1;
      if (rt < h_rpo || rt > cont_rpo) loop_merge_label[header] = tgt;
    }
  }

  // Make a helper to build a simple Instr with given opcode and operands.
  auto makeSynthInstr = [](uint32_t op, std::initializer_list<uint32_t> operands) -> Instr {
    Instr ins;
    ins.opcode = op;
    ins.words.push_back(0); // placeholder for opcode|wc
    for (uint32_t o : operands) ins.words.push_back(o);
    ins.words[0] = ((uint32_t)ins.words.size() << 16) | op;
    return ins;
  };

  // Reconstruct body with merge instructions injected before terminators.
  // Track which blocks already have merge instructions (from the original).
  std::vector<Instr> result;
  std::vector<uint32_t> synthetic_merges; // IDs of synthetic unreachable merge blocks
  for (auto& blk : blocks) {
    bool is_loop_hdr = loop_header_labels.count(blk.label_id) > 0;

    for (size_t k = 0; k < blk.instr_idx.size(); ++k) {
      const Instr& ins = body[blk.instr_idx[k]];
      bool is_last = (k + 1 == blk.instr_idx.size());

      if (is_last && isTerminator(ins.opcode)) {
        // Check whether the second-to-last instruction is already a merge.
        bool prev_is_merge = (k > 0 &&
            (body[blk.instr_idx[k-1]].opcode == Op::LoopMerge ||
             body[blk.instr_idx[k-1]].opcode == Op::SelectionMerge));
        if (!prev_is_merge) {
          if (is_loop_hdr && loop_merge_label.count(blk.label_id) &&
              loop_continue.count(blk.label_id)) {
            // Insert LoopMerge before the terminator.
            result.push_back(makeSynthInstr(Op::LoopMerge, {
              loop_merge_label[blk.label_id],
              loop_continue[blk.label_id],
              0 // loop control = None
            }));
          } else if (ins.opcode == Op::BranchConditional && ins.wc() >= 4) {
            uint32_t t1 = ins.word(2), t2 = ins.word(3);
            // Use empty stop set: BFS must be able to cross loop headers to find the
            // correct merge block for conditional branches that exit a loop.
            // (The !prev_is_merge check above already prevents adding SelectionMerge
            //  when the block already has LoopMerge.)
            uint32_t merge = findMerge(t1, t2, {});
            if (merge == 0 && next_id) {
              // Both branches return/unreachable — no common merge block exists.
              // Synthesize an unreachable merge block so the selection is structured.
              merge = (*next_id)++;
              synthetic_merges.push_back(merge);
            }
            if (merge != 0) {
              result.push_back(makeSynthInstr(Op::SelectionMerge, {merge, 0}));
            }
          } else if (ins.opcode == Op::Switch && ins.wc() >= 3) {
            // Find merge for switch: first block reachable from all targets.
            uint32_t dflt = ins.word(2);
            std::vector<uint32_t> targets = {dflt};
            for (uint32_t k2 = 3; k2 + 1 < ins.wc(); k2 += 2)
              targets.push_back(ins.word(k2 + 1));
            // Remove duplicates
            std::sort(targets.begin(), targets.end());
            targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
            if (targets.size() >= 2) {
              uint32_t merge = findMerge(targets[0], targets[1], {});
              for (size_t ti = 2; ti < targets.size() && merge != 0; ++ti)
                merge = findMerge(merge, targets[ti], {});
              if (merge == 0 && next_id) {
                merge = (*next_id)++;
                synthetic_merges.push_back(merge);
              }
              if (merge != 0)
                result.push_back(makeSynthInstr(Op::SelectionMerge, {merge, 0}));
            }
          }
        }
      }

      result.push_back(ins);
    }
  }

  // Append synthetic unreachable merge blocks (for selections where all branches return).
  for (uint32_t mid : synthetic_merges) {
    result.push_back(makeSynthInstr(Op::Label, {mid}));
    result.push_back(makeSynthInstr(Op::Unreachable, {}));
  }

  // Append FunctionEnd if present.
  for (auto& ins : body)
    if (ins.opcode == Op::FunctionEnd) { result.push_back(ins); break; }

  return result;
}

// ---------------------------------------------------------------------------
// ID allocator for Pass 2
// ---------------------------------------------------------------------------
struct IdAllocator {
  uint32_t next_id;
  explicit IdAllocator(uint32_t base) : next_id(base) {}
  uint32_t alloc() { return next_id++; }
};

// ---------------------------------------------------------------------------
// Pass 2: emit Vulkan PSB SPIR-V
// ---------------------------------------------------------------------------

static Words emitVulkanSpirv(ModuleInfo& info, std::string& err) {
  // Validate preconditions
  if (info.has_unsupported_globals) {
    err = "Module has non-chip_var CrossWorkgroup globals (e.g. __chipspv_device_heap); "
          "falling back to clvk's internal clspv pipeline";
    return {};
  }
  if (info.uint64_type_id == 0) {
    err = "No uint64 type found in module";
    return {};
  }
  if (info.uint32_type_id == 0) {
    err = "No uint32 type found in module";
    return {};
  }

  // Bail out for Intel-specific capabilities that require instruction-level translation
  // (not just capability substitution). clspv handles these via proper LLVM IR → SPIR-V
  // lowering. We cannot replicate that translation here.
  static const std::unordered_set<uint32_t> kIntelOnlyCaps = {
    5568u, // SubgroupShuffleINTEL       (OpSubgroupShuffleINTEL family)
    5569u, // SubgroupBufferBlockIOINTEL
    5570u, // SubgroupImageBlockIOINTEL
    5576u, // SubgroupImageMediaBlockIOINTEL
    5579u, // RoundToInfinityINTEL
    5580u, // FloatingPointModeINTEL
    5581u, // IntegerFunctions2INTEL
    5584u, // FunctionPointersINTEL
    5603u, // ExpectAssumeKHR
    5898u, // SubgroupAvcMotionEstimationINTEL
  };
  for (uint32_t cap : info.capabilities) {
    if (kIntelOnlyCaps.count(cap)) {
      // Return empty (no error) → silent fallback to clspv subprocess
      return {};
    }
  }

  // Check struct params for bad pointers
  for (auto fn_id : info.impl_fn_ids) {
    auto fit = info.functions.find(fn_id);
    if (fit == info.functions.end()) continue;
    for (auto& param : fit->second.params) {
      if (param.is_byval && param.byval_type_id != 0) {
        if (structHasBadPointers(param.byval_type_id, info)) {
          err = "ByVal struct param contains CrossWorkgroup/Generic pointer members — unsupported";
          return {};
        }
        // If the byval struct type is also used as a Function-scope pointer
        // base type, we would need to add Offset decorations to make it
        // Block-compatible — but VUID-10684 forbids Offset on structs used as
        // Function-scope variables.  Fall back to clspv subprocess silently.
        for (auto& [tid, pti] : info.ptr_types) {
          if (pti.base_type == param.byval_type_id && pti.storage_class == SC::Function) {
            return {}; // silent fallback
          }
        }
      }
    }
  }

  // If any function body uses OpExtInst with an import from the input module
  // (e.g. OpenCL.std fma/sqrt/etc.), we can't translate those to Vulkan.
  // Fall back to clspv subprocess silently.
  if (!info.ext_inst_import_ids.empty()) {
    for (auto& [fn_id, fn] : info.functions) {
      for (auto& instr : fn.body) {
        if (instr.opcode == Op::ExtInst && instr.wc() >= 4) {
          if (info.ext_inst_import_ids.count(instr.word(3)))
            return {}; // silent fallback — has OpenCL.std/GLSL calls
        }
      }
    }
  }

  IdAllocator idAlloc(info.bound + 1);

  // -------------------------------------------------------------------------
  // Allocate IDs for new constructs
  // -------------------------------------------------------------------------

  // WorkgroupSize spec constants and variable
  uint32_t wgs_x_id     = idAlloc.alloc();
  uint32_t wgs_y_id     = idAlloc.alloc();
  uint32_t wgs_z_id     = idAlloc.alloc();

  // v3uint type
  uint32_t v3uint_type_id = idAlloc.alloc();
  // spec constant composite for WGS (decorated with BuiltIn WorkgroupSize)
  uint32_t wgs_composite_id = idAlloc.alloc();
  // ptr Input v3uint — for BuiltIn WorkgroupId, LocalInvocationId, etc.
  uint32_t ptr_input_v3uint_id = idAlloc.alloc();

  // NonSemantic.ClspvReflection.5 ext import
  uint32_t clspv_refl_id = idAlloc.alloc();

  // For each chip_var: SSBO struct type, ptr type, variable
  for (auto& cv : info.chip_vars) {
    cv.ssbo_struct_type_id = idAlloc.alloc();
    cv.ssbo_ptr_type_id    = idAlloc.alloc();
    cv.ssbo_var_id         = idAlloc.alloc();
    cv.descriptor_set      = (uint32_t)(&cv - &info.chip_vars[0]);
    cv.binding             = 0;
  }

  // Map from base_type_id -> psb_ptr_type_id (ptr PSB <canonical_base>)
  // For scalar base: canonical_base = scalar type id
  // For CW/Generic ptr base: canonical_base = needPSBType(inner_base) (recursive)
  std::unordered_map<uint32_t, uint32_t> psb_ptr_type_map; // base_type_id -> psb_ptr_type_id
  // Maps each psb_ptr_id -> the canonical base type used in its OpTypePointer declaration
  std::unordered_map<uint32_t, uint32_t> psb_canonical_base_map;

  // Recursive lambda (needs std::function for self-reference).
  // needPSBType(X) returns the ID of "ptr PSB Y" where:
  //   - Y = needPSBType(X.base_type)  if X is a CW/Generic ptr type
  //   - Y = X                          otherwise (scalar / struct / etc.)
  std::function<uint32_t(uint32_t)> needPSBType = [&](uint32_t base_type_id) -> uint32_t {
    auto it = psb_ptr_type_map.find(base_type_id);
    if (it != psb_ptr_type_map.end()) return it->second;

    uint32_t canonical_base;
    auto pit = info.ptr_types.find(base_type_id);
    if (pit != info.ptr_types.end() &&
        (pit->second.storage_class == SC::CrossWorkgroup ||
         pit->second.storage_class == SC::Generic)) {
      // Recursive: ptr PSB (needPSBType(inner_base))
      canonical_base = needPSBType(pit->second.base_type);
    } else {
      canonical_base = base_type_id;
    }
    uint32_t new_id = idAlloc.alloc();
    psb_ptr_type_map[base_type_id] = new_id;
    psb_canonical_base_map[new_id] = canonical_base;
    return new_id;
  };

  // Pre-scan ALL function bodies to find all Generic/CW ptr types needed
  // (not just impl functions — entry points that fail stub detection also need PSB types)
  for (auto& [fn_id, fn_ref] : info.functions) {
    auto fit = info.functions.find(fn_id);
    if (fit == info.functions.end()) continue;
    FunctionInfo& fn = fit->second;

    // Params that are pointer kind need PSB type for their base type.
    // Also, non-stub non-impl helper functions pass CW ptr params as PSB ptrs,
    // so create PSB types for ALL their CW ptr params regardless of pointer kind.
    bool is_helper = !info.impl_fn_ids.count(fn_id) && !info.stub_fn_ids.count(fn_id);
    for (auto& p : fn.params) {
      if (p.is_crossworkgroup_ptr && (p.is_pointer_kind || is_helper)) {
        auto pit = info.ptr_types.find(p.type_id);
        if (pit != info.ptr_types.end())
          needPSBType(pit->second.base_type);
      }
    }

    // Scan body for ConvertUToPtr / PtrAccessChain / Bitcast to Generic/CW types
    for (auto& instr : fn.body) {
      if (instr.opcode == Op::ConvertUToPtr && instr.wc() >= 4) {
        uint32_t result_type = instr.word(1);
        auto pit = info.ptr_types.find(result_type);
        if (pit != info.ptr_types.end() &&
            (pit->second.storage_class == SC::Generic ||
             pit->second.storage_class == SC::CrossWorkgroup)) {
          needPSBType(pit->second.base_type);
        }
      }
      // InBoundsPtrAccessChain / PtrAccessChain with CW/Generic result type
      if ((instr.opcode == Op::InBoundsPtrAccessChain ||
           instr.opcode == Op::PtrAccessChain) && instr.wc() >= 5) {
        uint32_t result_type = instr.word(1);
        auto pit = info.ptr_types.find(result_type);
        if (pit != info.ptr_types.end() &&
            (pit->second.storage_class == SC::Generic ||
             pit->second.storage_class == SC::CrossWorkgroup)) {
          needPSBType(pit->second.base_type);
        }
      }
      // Bitcast to a Generic/CW pointer type (e.g. type-punning device pointers)
      if (instr.opcode == Op::Bitcast && instr.wc() >= 4) {
        uint32_t result_type = instr.word(1);
        auto pit = info.ptr_types.find(result_type);
        if (pit != info.ptr_types.end() &&
            (pit->second.storage_class == SC::Generic ||
             pit->second.storage_class == SC::CrossWorkgroup)) {
          needPSBType(pit->second.base_type);
        }
      }
    }
  }

  // Also need PSB ptr types for chip_var base types
  for (auto& cv : info.chip_vars) {
    if (cv.base_type_id != 0)
      needPSBType(cv.base_type_id);
  }

  // Push constant struct types: one per kernel entry point
  // Key: entry point fn_id (the stub), Value: {pc_struct_type_id, ptr_PC_type_id, pc_var_id, layout}
  struct KernelPCInfo {
    uint32_t pc_struct_type_id = 0;
    uint32_t ptr_pc_type_id    = 0;
    uint32_t pc_var_id         = 0;
    std::vector<ArgLayout> layout;
    // Per-member ptr types in PC struct (for ptr members)
    std::vector<uint32_t> member_ptr_type_ids; // ptr to each member type (for AccessChain)
    uint32_t total_size = 0;
  };
  std::unordered_map<uint32_t, KernelPCInfo> kernel_pc_info; // stub_fn_id -> KernelPCInfo

  for (uint32_t stub_id : info.stub_fn_ids) {
    auto sit = info.functions.find(stub_id);
    if (sit == info.functions.end()) continue;
    FunctionInfo& stub = sit->second;

    uint32_t impl_id = stub.impl_call_target;
    auto iit = info.functions.find(impl_id);
    if (iit == info.functions.end()) continue;
    FunctionInfo& impl_fn = iit->second;

    KernelPCInfo kpc;
    kpc.layout = computeKernelPCLayout(impl_fn, info);
    kpc.pc_struct_type_id = idAlloc.alloc();
    kpc.ptr_pc_type_id    = idAlloc.alloc();
    kpc.pc_var_id         = idAlloc.alloc();

    // Compute total size
    if (!kpc.layout.empty()) {
      auto& last = kpc.layout.back();
      kpc.total_size = last.pc_offset + last.pc_size;
    }

    // Pre-allocate ptr-to-member types for AccessChain
    for (auto& al : kpc.layout) {
      uint32_t ptr_id = idAlloc.alloc();
      kpc.member_ptr_type_ids.push_back(ptr_id);
    }

    kernel_pc_info[stub_id] = kpc;
  }

  // New function types needed: void() for each kernel stub
  // (impl fn types stay the same but with modified param types — need new IDs)
  // Actually for impl fns: params change from ptr CW T → ulong, so fn type changes
  // Map: impl_fn_id -> new fn_type_id
  std::unordered_map<uint32_t, uint32_t> impl_fn_new_type_ids;
  std::unordered_map<uint32_t, uint32_t> stub_fn_new_type_ids; // void() type

  // void() function type for stubs (all same)
  uint32_t void_fn_type_id = 0;
  // Check if one already exists
  for (auto& [fid, rt] : info.fn_types) {
    if (rt == info.void_type_id && info.fn_type_params[fid].empty()) {
      void_fn_type_id = fid;
      break;
    }
  }
  if (void_fn_type_id == 0)
    void_fn_type_id = idAlloc.alloc();

  // For each impl fn: compute new param types
  struct ImplFnNewParams {
    std::vector<uint32_t> new_param_type_ids; // same length as params
  };
  std::unordered_map<uint32_t, ImplFnNewParams> impl_fn_new_params;

  // Dedup new fn types: map from (ret_type, param_type...) → new type ID
  // to avoid emitting duplicate OpTypeFunction declarations.
  std::map<std::vector<uint32_t>, uint32_t> new_fn_type_by_sig;

  for (uint32_t fn_id : info.impl_fn_ids) {
    auto fit = info.functions.find(fn_id);
    if (fit == info.functions.end()) continue;
    FunctionInfo& fn = fit->second;

    ImplFnNewParams np;
    bool changed = false;
    for (auto& p : fn.params) {
      if (p.is_crossworkgroup_ptr) {
        np.new_param_type_ids.push_back(info.uint64_type_id);
        changed = true;
      } else {
        np.new_param_type_ids.push_back(p.type_id);
      }
    }
    impl_fn_new_params[fn_id] = np;
    if (changed) {
      // Build signature: (ret_type, param_types...)
      std::vector<uint32_t> sig = {fn.ret_type};
      for (uint32_t t : np.new_param_type_ids) sig.push_back(t);
      auto it = new_fn_type_by_sig.find(sig);
      if (it != new_fn_type_by_sig.end()) {
        impl_fn_new_type_ids[fn_id] = it->second; // reuse deduplicated ID
      } else {
        uint32_t new_id = idAlloc.alloc();
        impl_fn_new_type_ids[fn_id] = new_id;
        new_fn_type_by_sig[sig] = new_id;
      }
    } else {
      impl_fn_new_type_ids[fn_id] = fn.fn_type;
    }
  }

  // Constants we need: 0, 1, 2 as uint32
  // Check if they exist
  uint32_t const_uint32_0_id = 0;
  uint32_t const_uint32_1_id = 0;
  uint32_t const_uint32_2_id = 0;
  for (auto& [cid, ci] : info.constants) {
    if (ci.type_id == info.uint32_type_id) {
      if (ci.value == 0 && const_uint32_0_id == 0) const_uint32_0_id = cid;
      if (ci.value == 1 && const_uint32_1_id == 0) const_uint32_1_id = cid;
      if (ci.value == 2 && const_uint32_2_id == 0) const_uint32_2_id = cid;
    }
  }
  if (const_uint32_0_id == 0) const_uint32_0_id = idAlloc.alloc();
  if (const_uint32_1_id == 0) const_uint32_1_id = idAlloc.alloc();
  if (const_uint32_2_id == 0) const_uint32_2_id = idAlloc.alloc();

  // Function-scope ptr types for ByVal structs in kernel stubs
  // Map: byval_struct_type_id -> ptr_Function_struct_type_id
  // Prefer reusing existing OpTypePointer Function %struct from the module to avoid
  // duplicate type declarations (which cause OpFunctionCall type mismatch errors).
  std::unordered_map<uint32_t, uint32_t> ptr_function_struct_type_map;
  // Build reverse map: (SC::Function, base_type) -> existing type_id
  std::unordered_map<uint32_t, uint32_t> existing_fn_ptr_types; // base_type -> type_id
  for (auto& [tid, pti] : info.ptr_types) {
    if (pti.storage_class == SC::Function)
      existing_fn_ptr_types[pti.base_type] = tid;
  }
  for (auto& [stub_id, kpc] : kernel_pc_info) {
    for (auto& al : kpc.layout) {
      if (al.is_byval) {
        auto it = ptr_function_struct_type_map.find(al.type_id);
        if (it == ptr_function_struct_type_map.end()) {
          // Reuse existing ptr Function type if available, else allocate new.
          auto eit = existing_fn_ptr_types.find(al.type_id);
          ptr_function_struct_type_map[al.type_id] =
              (eit != existing_fn_ptr_types.end()) ? eit->second : idAlloc.alloc();
        }
      }
    }
  }

  // Build a map: CrossWorkgroup/Generic ptr type ID → PSB ptr type ID.
  // Used when emitting function bodies so any reference to a filtered CW/Generic
  // ptr type is replaced with the corresponding PSB ptr type.
  std::unordered_map<uint32_t, uint32_t> cw_type_remap;
  for (auto& [tid, pti] : info.ptr_types) {
    if (pti.storage_class == SC::CrossWorkgroup || pti.storage_class == SC::Generic) {
      auto it = psb_ptr_type_map.find(pti.base_type);
      if (it != psb_ptr_type_map.end())
        cw_type_remap[tid] = it->second;
    }
  }

  // For non-stub, non-impl ("helper") functions that have CW/Generic ptr params:
  // compute new fn_types where CW ptr params → PSB ptr params.
  std::unordered_map<uint32_t, uint32_t> helper_fn_new_type_ids;
  std::unordered_map<uint32_t, ImplFnNewParams> helper_fn_new_params;
  for (auto& [fn_id, fn] : info.functions) {
    if (info.impl_fn_ids.count(fn_id) || info.stub_fn_ids.count(fn_id)) continue;
    ImplFnNewParams np;
    bool changed = false;
    for (auto& p : fn.params) {
      auto cwit = cw_type_remap.find(p.type_id);
      if (cwit != cw_type_remap.end()) {
        np.new_param_type_ids.push_back(cwit->second); // CW/Generic ptr → PSB ptr
        changed = true;
      } else {
        np.new_param_type_ids.push_back(p.type_id);
      }
    }
    if (changed) {
      std::vector<uint32_t> sig = {fn.ret_type};
      for (uint32_t t : np.new_param_type_ids) sig.push_back(t);
      auto it = new_fn_type_by_sig.find(sig);
      if (it != new_fn_type_by_sig.end()) {
        helper_fn_new_type_ids[fn_id] = it->second;
      } else {
        uint32_t new_id = idAlloc.alloc();
        helper_fn_new_type_ids[fn_id] = new_id;
        new_fn_type_by_sig[sig] = new_id;
      }
      helper_fn_new_params[fn_id] = np;
    }
  }

  // -------------------------------------------------------------------------
  // Pre-collect all strings needed for NonSemantic.ClspvReflection metadata.
  // Strings must be emitted as OpString in the debug section (before type
  // definitions), not inline in the reflection body.
  // -------------------------------------------------------------------------
  std::unordered_map<std::string, uint32_t> string_ids;
  auto allocStringId = [&](const std::string& s) -> uint32_t {
    auto it = string_ids.find(s);
    if (it != string_ids.end()) return it->second;
    uint32_t id = idAlloc.alloc();
    string_ids[s] = id;
    return id;
  };

  for (auto& ep : info.entry_points) {
    auto kpc_it = kernel_pc_info.find(ep.fn_id);
    if (kpc_it == kernel_pc_info.end()) continue;
    KernelPCInfo& kpc = kpc_it->second;

    auto sit = info.functions.find(ep.fn_id);
    if (sit == info.functions.end()) continue;
    uint32_t impl_id = sit->second.impl_call_target;
    auto iit = info.functions.find(impl_id);
    if (iit == info.functions.end()) continue;
    FunctionInfo& impl_fn = iit->second;

    allocStringId(ep.name);
    allocStringId(""); // empty attr string
    for (size_t i = 0; i < kpc.layout.size(); ++i) {
      std::string arg_name;
      if (i < impl_fn.params.size()) {
        auto nit = info.names.find(impl_fn.params[i].id);
        if (nit != info.names.end()) arg_name = nit->second;
      }
      allocStringId(arg_name);
    }
  }
  for (auto& cv : info.chip_vars) {
    uint32_t type_bytes = 8;
    auto wit = info.int_widths.find(cv.base_type_id);
    if (wit != info.int_widths.end()) type_bytes = (wit->second + 7) / 8;
    else {
      auto fit = info.float_widths.find(cv.base_type_id);
      if (fit != info.float_widths.end()) type_bytes = (fit->second + 7) / 8;
    }
    allocStringId(std::string(type_bytes * 2, '0'));
  }

  // -------------------------------------------------------------------------
  // Pre-collect all uint32 constants needed for reflection metadata.
  // Constants must be in the globals section (before functions).
  // We maintain a map of value → constant_id, reusing existing constants
  // from info.constants where possible.
  // -------------------------------------------------------------------------
  std::unordered_map<uint32_t, uint32_t> refl_const_ids; // value -> id
  auto getReflConst = [&](uint32_t value) -> uint32_t {
    auto it = refl_const_ids.find(value);
    if (it != refl_const_ids.end()) return it->second;
    // Reuse existing constant if available
    for (auto& [cid, ci] : info.constants) {
      if (ci.type_id == info.uint32_type_id && ci.value == value) {
        refl_const_ids[value] = cid;
        return cid;
      }
    }
    uint32_t id = idAlloc.alloc();
    refl_const_ids[value] = id;
    return id;
  };

  // Pre-allocate all reflection constants
  getReflConst(0); getReflConst(1); getReflConst(2); // specid constants
  for (auto& ep : info.entry_points) {
    auto kpc_it = kernel_pc_info.find(ep.fn_id);
    if (kpc_it == kernel_pc_info.end()) continue;
    KernelPCInfo& kpc = kpc_it->second;
    getReflConst((uint32_t)kpc.layout.size()); // num_args
    for (size_t i = 0; i < kpc.layout.size(); ++i) {
      getReflConst((uint32_t)i);               // ordinal
      getReflConst(kpc.layout[i].pc_offset);   // offset
      getReflConst(kpc.layout[i].pc_size);     // size
    }
  }
  for (auto& cv : info.chip_vars) {
    getReflConst(cv.descriptor_set);
    getReflConst(cv.binding);
  }

  // -------------------------------------------------------------------------
  // Now compute the final new bound
  // -------------------------------------------------------------------------
  uint32_t new_bound = idAlloc.next_id;

  // -------------------------------------------------------------------------
  // Emit output SPIR-V
  // -------------------------------------------------------------------------
  Words out;

  // --- Header ---
  // LocalSizeId (used for WGS spec constants) requires SPIR-V 1.2+.
  // NonSemantic extensions require 1.5+. Emit at least 1.6 (Vulkan 1.3).
  static const uint32_t kMinVersion = 0x00010600u; // SPIR-V 1.6
  out.push_back(spv::MagicNumber);
  out.push_back(std::max(info.version, kMinVersion));
  out.push_back(0x00000000); // generator = 0 (or clspv-like)
  out.push_back(new_bound);
  out.push_back(0); // schema

  // --- Capabilities ---
  // Always emit Vulkan-required capabilities.
  emitInstr(out, Op::Capability, {Cap::Shader});
  emitInstr(out, Op::Capability, {Cap::VariablePointers});
  emitInstr(out, Op::Capability, {Cap::PhysicalStorageBufferAddresses});

  // Forward all original capabilities except OpenCL-specific ones.
  // OpenCL-specific (not valid in Vulkan): Kernel(6), Addresses(4), Linkage(5),
  // GenericPointer(38), SubgroupDispatch(50), NamedBarrier(51), PipeStorage(52),
  // Float16Buffer(48).
  // Also skip capabilities we already emitted above to avoid duplicates.
  {
    static const std::unordered_set<uint32_t> kDropCaps = {
      Cap::Kernel, Cap::Addresses, Cap::Linkage, Cap::GenericPointer,
      Cap::Shader, Cap::VariablePointers, Cap::PhysicalStorageBufferAddresses,
      7u,  // Vector16 (OpenCL Kernel only)
      8u,  // Float16Buffer (OpenCL Kernel only)
      58u, // SubgroupDispatch (OpenCL Kernel only)
      59u, // NamedBarrier (OpenCL Kernel only)
      60u, // PipeStorage (OpenCL Pipe only)
    };
    for (uint32_t cap : info.capabilities) {
      if (kDropCaps.count(cap)) continue;
      emitInstr(out, Op::Capability, {cap});
    }
  }
  // SubgroupDispatch (58) is OpenCL-only; replace with GroupNonUniform (61)
  // so SubgroupLocalInvocationId builtin decoration remains valid in Vulkan 1.1+.
  if (info.capabilities.count(58u)) {
    emitInstr(out, Op::Capability, {61u}); // GroupNonUniform (SPIR-V 1.3 core)
  }
  // Float16Buffer (8) is OpenCL-only; replace with Float16 (9) for Vulkan.
  if (info.capabilities.count(8u) && !info.capabilities.count(Cap::Float16)) {
    emitInstr(out, Op::Capability, {Cap::Float16}); // Float16
  }

  // --- Extensions ---
  std::unordered_set<std::string> emitted_exts;
  auto emitExtension = [&](const std::string& name) {
    if (!emitted_exts.insert(name).second) return; // deduplicate
    Words ops;
    Words strWords = encodeLiteralString(name);
    for (uint32_t w : strWords) ops.push_back(w);
    emitInstr(out, Op::Extension, ops);
  };
  emitExtension("SPV_KHR_non_semantic_info");
  emitExtension("SPV_KHR_storage_buffer_storage_class");
  emitExtension("SPV_KHR_variable_pointers");
  emitExtension("SPV_KHR_physical_storage_buffer");

  // Forward original extensions, dropping OpenCL-only ones.
  for (const auto& ext : info.extensions) {
    // Drop OpenCL extensions (cl_khr_*, cl_intel_*) — these are not Vulkan
    if (ext.find("cl_khr_") == 0) continue;
    if (ext.find("cl_intel_") == 0) continue;
    emitExtension(ext); // deduplication is handled inside
  }

  // --- OpExtInstImport NonSemantic.ClspvReflection.5 ---
  {
    Words ops = {clspv_refl_id};
    Words strWords = encodeLiteralString("NonSemantic.ClspvReflection.5");
    for (uint32_t w : strWords) ops.push_back(w);
    emitInstr(out, Op::ExtInstImport, ops);
  }

  // --- OpMemoryModel PhysicalStorageBuffer64 GLSL450 ---
  emitInstr(out, Op::MemoryModel, {
    spv::AddressingModelPhysicalStorageBuffer64,
    spv::MemoryModelGLSL450
  });

  // --- OpEntryPoint GLCompute ---
  for (auto& ep : info.entry_points) {
    Words ops;
    ops.push_back(spv::ExecutionModelGLCompute);
    ops.push_back(ep.fn_id);
    Words strWords = encodeLiteralString(ep.name);
    for (uint32_t w : strWords) ops.push_back(w);
    // Interface vars (SPIR-V 1.4+: must include all statically-used global vars)
    // Keep original interface vars that are still valid (filter removed IDs)
    for (uint32_t iv : ep.interface_ids) {
      if (info.chip_var_by_id.count(iv)) continue; // replaced by SSBOs
      if (iv == info.workgroup_size_var_id) continue; // replaced by spec constant composite
      ops.push_back(iv);
    }
    // Add all chip_var SSBO vars (used transitively by most entry points)
    for (auto& cv : info.chip_vars) {
      ops.push_back(cv.ssbo_var_id);
    }
    // Add the PC var for this stub entry point (if it has one)
    auto kpc_it = kernel_pc_info.find(ep.fn_id);
    if (kpc_it != kernel_pc_info.end())
      ops.push_back(kpc_it->second.pc_var_id);
    emitInstr(out, Op::EntryPoint, ops);
  }

  // WorkgroupSize is specified via BuiltIn WorkgroupSize on the SpecConstantComposite (clspv style).

  // --- OpSource ---
  // Emit a simple source annotation
  emitInstr(out, Op::Source, {
    (uint32_t)spv::SourceLanguageOpenCL_C,
    120 // version 1.2
  });

  // --- Debug section ---
  // SPIR-V layout requires: OpString → OpName/OpMemberName (in that order).
  // Emit all OpString instructions first (from original module + reflection),
  // then all OpName / OpMemberName instructions.

  // Pass 1: emit all OpString instructions from original module
  for (auto& instr : info.debug_instrs) {
    if (instr.opcode == Op::String)
      emit(out, instr.words);
  }
  // Emit new OpString instructions for NonSemantic.ClspvReflection strings
  for (auto& [s, id] : string_ids) {
    Words ops = {id};
    Words strWords = encodeLiteralString(s);
    for (uint32_t w : strWords) ops.push_back(w);
    emitInstr(out, Op::String, ops);
  }

  // Build set of IDs that are removed/replaced in the output and must not appear in OpName.
  // Includes: chip_var IDs (replaced by SSBO vars) and original stub parameter IDs
  // (stub bodies are completely replaced with new bodies that have no original params).
  std::unordered_set<uint32_t> removed_ids;
  for (auto& [id, idx] : info.chip_var_by_id)
    removed_ids.insert(id);
  for (auto& [id, _] : info.init_var_constants)
    removed_ids.insert(id);
  // WorkgroupSize variable is removed (replaced by spec constant composite)
  if (info.workgroup_size_var_id != 0)
    removed_ids.insert(info.workgroup_size_var_id);
  for (uint32_t stub_id : info.stub_fn_ids) {
    auto it = info.functions.find(stub_id);
    if (it == info.functions.end()) continue;
    for (auto& p : it->second.params)
      removed_ids.insert(p.id);
  }

  // Pass 2: emit all OpName / OpMemberName instructions from original module,
  // skipping those that reference IDs not present in the output.
  for (auto& instr : info.debug_instrs) {
    if (instr.opcode == Op::String) continue; // already emitted
    if (instr.opcode == Op::Name && instr.wc() >= 3) {
      if (removed_ids.count(instr.word(1))) continue;
    }
    emit(out, instr.words);
  }
  // Add names for new SSBO vars
  for (auto& cv : info.chip_vars) {
    Words ops = {cv.ssbo_var_id};
    Words strWords = encodeLiteralString(cv.name);
    for (uint32_t w : strWords) ops.push_back(w);
    emitInstr(out, Op::Name, ops);
  }

  // --- OpDecorate ---

  // Block decoration for chip_var SSBO struct types
  for (auto& cv : info.chip_vars) {
    emitInstr(out, Op::Decorate, {cv.ssbo_struct_type_id, Deco::Block});
  }
  // DescriptorSet / Binding for chip_var SSBO vars
  for (auto& cv : info.chip_vars) {
    emitInstr(out, Op::Decorate, {cv.ssbo_var_id, Deco::DescriptorSet, cv.descriptor_set});
    emitInstr(out, Op::Decorate, {cv.ssbo_var_id, Deco::Binding, cv.binding});
  }

  // Block + DescriptorSet/Binding for PC structs
  // Note: PushConstant doesn't use DescriptorSet/Binding, but needs Block
  for (auto& [stub_id, kpc] : kernel_pc_info) {
    emitInstr(out, Op::Decorate, {kpc.pc_struct_type_id, Deco::Block});
  }

  // Member offsets for PC struct members
  for (auto& [stub_id, kpc] : kernel_pc_info) {
    for (size_t i = 0; i < kpc.layout.size(); ++i) {
      emitInstr(out, Op::MemberDecorate, {
        kpc.pc_struct_type_id, (uint32_t)i, Deco::Offset, kpc.layout[i].pc_offset
      });
    }
  }

  // Member offsets for chip_var SSBO struct types (single member at offset 0)
  for (auto& cv : info.chip_vars) {
    emitInstr(out, Op::MemberDecorate, {
      cv.ssbo_struct_type_id, 0, Deco::Offset, 0
    });
  }

  // Pass through member decorations from the input module (Offset, BuiltIn, etc.)
  // and ensure all nested struct types reachable from Block-decorated PC structs
  // have Offset decorations (Vulkan requires explicit layout for all nested structs).
  {
    // Track struct IDs whose Offset decorations are already emitted
    std::unordered_set<uint32_t> offset_emitted_structs;
    for (auto& [stub_id, kpc] : kernel_pc_info)
      offset_emitted_structs.insert(kpc.pc_struct_type_id);
    for (auto& cv : info.chip_vars)
      offset_emitted_structs.insert(cv.ssbo_struct_type_id);

    // Emit passthrough member decorations, skipping OpenCL-specific and
    // Offset decorations for structs we already handled above.
    static const uint32_t DecoFuncParamAttr    = 38;
    static const uint32_t DecoMemberAlignment  = 44;
    static const uint32_t DecoMemberMaxByteOff = 45;
    static const uint32_t DecoMemberLinkage    = 41;
    for (auto& [struct_id, midx_map] : info.member_decorations) {
      for (auto& [member_idx, decos] : midx_map) {
        for (auto& deco : decos) {
          if (deco.empty()) continue;
          uint32_t dk = deco[0];
          if (dk == DecoFuncParamAttr || dk == DecoMemberAlignment ||
              dk == DecoMemberMaxByteOff || dk == DecoMemberLinkage) continue;
          // Skip Offset for structs already handled to avoid duplicates
          if (dk == Deco::Offset && offset_emitted_structs.count(struct_id)) continue;
          Words ops = {struct_id, member_idx};
          for (uint32_t w : deco) ops.push_back(w);
          emitInstr(out, Op::MemberDecorate, ops);
          if (dk == Deco::Offset) offset_emitted_structs.insert(struct_id);
        }
      }
    }

    // For any nested struct reachable from a PC Block struct that still lacks
    // Offset decorations (not in input), compute them from type layout.
    // Note: kpc.pc_struct_type_id is a NEW ID not in info.struct_members,
    // so we start from the member types of kpc.layout directly.
    std::unordered_set<uint32_t> visited;
    // ensureStructOffsets: emit Offset decorations for all members of struct
    // type_id (if not already done), then recurse into nested struct members.
    std::function<void(uint32_t)> ensureStructOffsets = [&](uint32_t type_id) {
      if (visited.count(type_id)) return;
      visited.insert(type_id);
      auto sit = info.struct_members.find(type_id);
      if (sit == info.struct_members.end()) return;
      const auto& members = sit->second;

      // Emit Offset decorations for this struct's members if not already done
      if (!offset_emitted_structs.count(type_id)) {
        offset_emitted_structs.insert(type_id);
        uint32_t off = 0;
        for (size_t j = 0; j < members.size(); ++j) {
          uint32_t mt = members[j];
          uint32_t al = typeAlign(mt, info);
          if (al < 1) al = 1;
          off = (off + al - 1) & ~(al - 1);
          // Use input-provided offset if available (more accurate)
          uint32_t actual_off = off;
          auto omit = info.member_offsets.find(type_id);
          if (omit != info.member_offsets.end()) {
            auto oit = omit->second.find((uint32_t)j);
            if (oit != omit->second.end()) actual_off = oit->second;
          }
          emitInstr(out, Op::MemberDecorate, {type_id, (uint32_t)j, Deco::Offset, actual_off});
          off += typeSize(mt, info);
        }
      }

      // Recurse into nested struct members
      for (uint32_t mtype : members) {
        if (info.struct_members.count(mtype))
          ensureStructOffsets(mtype);
      }
    };
    // Start from each layout member type (not kpc.pc_struct_type_id which is new)
    for (auto& [stub_id, kpc] : kernel_pc_info) {
      for (auto& al : kpc.layout) {
        if (info.struct_members.count(al.type_id))
          ensureStructOffsets(al.type_id);
      }
    }
    // Also ensure PSB pointer target structs have Offset decorations.
    // Vulkan validators require explicit layout for structs referenced via
    // PhysicalStorageBuffer pointers (treated as Block-like).
    for (auto& [base_type_id, psb_ptr_id] : psb_ptr_type_map) {
      if (info.struct_members.count(base_type_id))
        ensureStructOffsets(base_type_id);
    }
  }

  // SpecId decorations for WGS spec constants
  emitInstr(out, Op::Decorate, {wgs_x_id, Deco::SpecId, 0});
  emitInstr(out, Op::Decorate, {wgs_y_id, Deco::SpecId, 1});
  emitInstr(out, Op::Decorate, {wgs_z_id, Deco::SpecId, 2});

  // BuiltIn WorkgroupSize on the SpecConstantComposite (clspv style)
  emitInstr(out, Op::Decorate, {wgs_composite_id, Deco::BuiltIn, Builtin::WorkgroupSize});

  // ArrayStride for PSB pointer types used in OpPtrAccessChain.
  // SPIR-V requires ArrayStride on any PSB pointer type that is the Base of OpPtrAccessChain.
  // Use psb_canonical_base_map to get each PSB type's actual declared base.
  {
    std::unordered_set<uint32_t> psb_ids;
    for (auto& [base, psb_id] : psb_ptr_type_map) psb_ids.insert(psb_id);
    for (auto& [psb_ptr_id, canonical_base] : psb_canonical_base_map) {
      uint32_t stride = 0;
      if (psb_ids.count(canonical_base)) {
        stride = 8; // canonical_base is itself a PSB ptr — 64-bit physical address
      } else {
        stride = typeSize(canonical_base, info);
      }
      if (stride > 0)
        emitInstr(out, Op::Decorate, {psb_ptr_id, Deco::ArrayStride, stride});
    }
  }

  // Emit original decorations from the input module, filtering OpenCL-specific ones.
  // Key decorations to emit: BuiltIn WorkgroupId, LocalInvocationId, GlobalInvocationId, etc.
  // Key decorations to DROP:
  //   LinkageAttributes (41) — requires Linkage capability, invalid in Shader
  //   Constant (22) — requires Kernel capability, invalid in Shader
  //   FuncParamAttr (38) — requires Kernel capability
  //   BuiltIn WorkgroupSize (25) on any variable — our SpecConstantComposite handles it
  {
    static const uint32_t DecoLinkageAttributes   = 41;
    static const uint32_t DecoConstantDecoration  = 22;
    static const uint32_t DecoFuncParamAttr       = 38;
    static const uint32_t DecoAlignmentAttr       = 44; // Alignment — requires Kernel capability
    static const uint32_t DecoMaxByteOffset       = 45; // MaxByteOffset — requires Kernel capability
    static const uint32_t BuiltInWorkgroupSize    = 25;

    for (auto& [id, deco_list] : info.decorations) {
      if (info.chip_var_by_id.count(id)) continue; // chip_var IDs — replaced
      if (id == info.workgroup_size_var_id) continue; // removed variable

      for (auto& deco : deco_list) {
        if (deco.empty()) continue;
        uint32_t deco_kind = deco[0];

        if (deco_kind == DecoLinkageAttributes) continue;
        if (deco_kind == DecoConstantDecoration) continue;
        if (deco_kind == DecoFuncParamAttr) continue;
        if (deco_kind == DecoAlignmentAttr) continue;
        if (deco_kind == DecoMaxByteOffset) continue;
        // Drop BuiltIn WorkgroupSize on any variable (spec constant handles it)
        if (deco_kind == Deco::BuiltIn && deco.size() >= 2 && deco[1] == BuiltInWorkgroupSize)
          continue;

        Words ops = {id};
        for (uint32_t w : deco) ops.push_back(w);
        emitInstr(out, Op::Decorate, ops);
      }
    }
  }

  // --- Type definitions ---

  // We skip CrossWorkgroup and Generic pointer types from the original.
  // We skip original function types for impl fns (we'll emit new ones).
  // We skip original function types for stubs.
  std::unordered_set<uint32_t> skip_types;

  // Mark CW/Generic ptr types to skip
  for (auto& [tid, pti] : info.ptr_types) {
    if (pti.storage_class == SC::CrossWorkgroup || pti.storage_class == SC::Generic)
      skip_types.insert(tid);
  }
  // Mark impl fn type IDs to skip (we'll emit new ones)
  for (auto& [fn_id, new_type_id] : impl_fn_new_type_ids) {
    auto fit = info.functions.find(fn_id);
    if (fit == info.functions.end()) continue;
    if (new_type_id != fit->second.fn_type)
      skip_types.insert(fit->second.fn_type);
  }
  // Mark helper fn type IDs to skip (we'll emit new ones with PSB ptr params)
  for (auto& [fn_id, new_type_id] : helper_fn_new_type_ids) {
    auto fit = info.functions.find(fn_id);
    if (fit == info.functions.end()) continue;
    skip_types.insert(fit->second.fn_type);
  }

  // Emit original types (filtering skipped)
  for (auto& instr : info.original_types) {
    uint32_t result_id = 0;
    if (instr.wc() >= 2) result_id = instr.word(1);
    if (skip_types.count(result_id)) continue;
    if (instr.opcode == Op::Variable) {
      uint32_t id = instr.word(2);
      if (info.chip_var_by_id.count(id)) continue; // chip_var replaced by SSBO
      if (info.builtin_input_vars.count(id)) continue; // BuiltIn Input — emitted later with correct type
    }
    // Transform SequentiallyConsistent memory semantics constants to AcquireRelease.
    // In Vulkan, SequentiallyConsistent (0x10) is not allowed; use AcquireRelease (0x8).
    // A constant is a memory semantics value if it has SequentiallyConsistent (0x10) AND
    // at least one memory-scope bit (UniformMemory=0x40, SubgroupMemory=0x80,
    // WorkgroupMemory=0x100, CrossWorkgroupMemory=0x200, AtomicCounterMemory=0x400,
    // ImageMemory=0x800).
    if (instr.opcode == Op::Constant && instr.wc() >= 4) {
      uint32_t type_id  = instr.word(1);
      uint32_t const_id = instr.word(2);
      uint32_t value    = instr.word(3);
      bool is_uint32 = (info.int_widths.count(type_id) && info.int_widths.at(type_id) == 32);
      if (is_uint32 && (value & 0x10u) && (value & 0xFC0u)) {
        // Replace SequentiallyConsistent (0x10) with AcquireRelease (0x8)
        uint32_t fixed = (value & ~0x10u) | 0x8u;
        emitInstr(out, Op::Constant, {type_id, const_id, fixed});
        continue;
      }
    }
    emit(out, instr.words);
  }

  // New: v3uint type
  emitInstr(out, Op::TypeVector, {v3uint_type_id, info.uint32_type_id, 3});

  // New: ptr Input v3uint — for BuiltIn WorkgroupId, LocalInvocationId, etc.
  emitInstr(out, Op::TypePointer, {ptr_input_v3uint_id, SC::Input, v3uint_type_id});

  // Emit BuiltIn Input variables with corrected Vulkan types.
  // Vector builtins (GlobalInvocationId, WorkgroupId, LocalInvocationId) need v3ulong→v3uint.
  // Scalar builtins (SubgroupLocalInvocationId, SubgroupSize, etc.) keep their original type.
  // The WorkgroupSize variable is intentionally omitted (replaced by SpecConstantComposite).
  for (auto& [var_id, builtin_kind] : info.builtin_input_vars) {
    if (builtin_kind == 25 /*WorkgroupSize*/) continue; // replaced by spec const
    // Determine the correct pointer type for this variable.
    uint32_t orig_ptr_type = 0;
    {
      auto oit = info.builtin_input_ptr_types.find(var_id);
      if (oit != info.builtin_input_ptr_types.end())
        orig_ptr_type = oit->second;
    }
    // Check if the original base type is a vector (needs v3ulong→v3uint replacement)
    // or a scalar (keep original type as-is).
    bool is_vector_builtin = false;
    if (orig_ptr_type) {
      auto pit = info.ptr_types.find(orig_ptr_type);
      if (pit != info.ptr_types.end()) {
        // If base type is a vector, it's a vector builtin needing replacement
        is_vector_builtin = (info.vec_types.count(pit->second.base_type) > 0);
      }
    }
    if (is_vector_builtin) {
      emitInstr(out, Op::Variable, {ptr_input_v3uint_id, var_id, SC::Input});
    } else {
      // Scalar builtin: keep original pointer type (it's already correct for Vulkan)
      emitInstr(out, Op::Variable, {orig_ptr_type, var_id, SC::Input});
    }
  }

  // New: PSB pointer types emitted in topological order (inner types before outer).
  // psb_canonical_base_map[psb_id] = canonical_base used in OpTypePointer declaration.
  // If canonical_base is itself a PSB type, it must be emitted first.
  {
    std::unordered_set<uint32_t> psb_id_set;
    for (auto& [base, psb_id] : psb_ptr_type_map) psb_id_set.insert(psb_id);

    std::unordered_set<uint32_t> emitted_psb;
    std::function<void(uint32_t)> emitPSBType = [&](uint32_t psb_id) {
      if (emitted_psb.count(psb_id)) return;
      auto cbit = psb_canonical_base_map.find(psb_id);
      if (cbit == psb_canonical_base_map.end()) return; // unknown — skip
      uint32_t canonical_base = cbit->second;
      // Emit dependency first (if canonical_base is itself a PSB type)
      if (psb_id_set.count(canonical_base))
        emitPSBType(canonical_base);
      emitted_psb.insert(psb_id);
      emitInstr(out, Op::TypePointer, {psb_id, SC::PhysicalStorageBuffer, canonical_base});
    };
    for (auto& [base, psb_id] : psb_ptr_type_map)
      emitPSBType(psb_id);
  }

  // New: SSBO struct types for chip_vars
  for (auto& cv : info.chip_vars) {
    emitInstr(out, Op::TypeStruct, {cv.ssbo_struct_type_id, cv.base_type_id});
    emitInstr(out, Op::TypePointer, {cv.ssbo_ptr_type_id, SC::StorageBuffer, cv.ssbo_struct_type_id});
  }

  // Also: StorageBuffer ptr to the base type of each chip_var (for AccessChain result)
  // These are needed for the transformed bodies
  // We already have psb_ptr_type_map for the base types; but for SSBO we need ptr StorageBuffer T
  // Actually for chip_var access we use AccessChain on the SSBO struct → gets a ptr StorageBuffer base_type
  // We need that type; create it per chip_var base type
  std::unordered_map<uint32_t, uint32_t> ptr_sb_type_map; // base_type -> ptr StorageBuffer type id
  for (auto& cv : info.chip_vars) {
    if (!ptr_sb_type_map.count(cv.base_type_id)) {
      uint32_t new_id = idAlloc.alloc();
      ptr_sb_type_map[cv.base_type_id] = new_id;
      // Emit now
      emitInstr(out, Op::TypePointer, {new_id, SC::StorageBuffer, cv.base_type_id});
    }
  }

  // New: PC struct types and ptr PC types
  for (auto& [stub_id, kpc] : kernel_pc_info) {
    // PC struct: one member per arg
    Words pc_struct_words = {kpc.pc_struct_type_id};
    for (auto& al : kpc.layout) {
      pc_struct_words.push_back(al.type_id);
    }
    emitInstr(out, Op::TypeStruct, pc_struct_words);

    // ptr PushConstant to PC struct
    emitInstr(out, Op::TypePointer, {kpc.ptr_pc_type_id, SC::PushConstant, kpc.pc_struct_type_id});

    // ptr PushConstant to each member type (for OpAccessChain)
    for (size_t i = 0; i < kpc.layout.size(); ++i) {
      emitInstr(out, Op::TypePointer, {kpc.member_ptr_type_ids[i], SC::PushConstant, kpc.layout[i].type_id});
    }
  }

  // New: ptr Function types for ByVal struct locals (only when not already in module)
  for (auto& [struct_type, ptr_fn_type] : ptr_function_struct_type_map) {
    auto eit = existing_fn_ptr_types.find(struct_type);
    if (eit != existing_fn_ptr_types.end() && eit->second == ptr_fn_type)
      continue; // already in the module, will be emitted from original_types
    emitInstr(out, Op::TypePointer, {ptr_fn_type, SC::Function, struct_type});
  }

  // New: impl fn type IDs (when params changed) — deduplicated
  std::unordered_set<uint32_t> emitted_fn_types;
  for (auto& [fn_id, new_type_id] : impl_fn_new_type_ids) {
    auto fit = info.functions.find(fn_id);
    if (fit == info.functions.end()) continue;
    if (new_type_id == fit->second.fn_type) continue; // unchanged
    if (emitted_fn_types.count(new_type_id)) continue; // already emitted
    emitted_fn_types.insert(new_type_id);
    FunctionInfo& fn = fit->second;
    auto& np = impl_fn_new_params[fn_id];
    Words fn_type_words = {new_type_id, fn.ret_type};
    for (uint32_t pt : np.new_param_type_ids)
      fn_type_words.push_back(pt);
    emitInstr(out, Op::TypeFunction, fn_type_words);
  }

  // New: helper fn type IDs (non-stub, non-impl fns with CW/Generic ptr params → PSB ptr)
  for (auto& [fn_id, new_type_id] : helper_fn_new_type_ids) {
    if (emitted_fn_types.count(new_type_id)) continue;
    emitted_fn_types.insert(new_type_id);
    auto fit = info.functions.find(fn_id);
    if (fit == info.functions.end()) continue;
    FunctionInfo& fn = fit->second;
    auto& np = helper_fn_new_params[fn_id];
    Words fn_type_words = {new_type_id, fn.ret_type};
    for (uint32_t pt : np.new_param_type_ids)
      fn_type_words.push_back(pt);
    emitInstr(out, Op::TypeFunction, fn_type_words);
  }

  // New: void() type for stubs (if not already present)
  {
    bool already_exists = false;
    for (auto& [tid, rt] : info.fn_types) {
      if (tid == void_fn_type_id) { already_exists = true; break; }
    }
    if (!already_exists) {
      emitInstr(out, Op::TypeFunction, {void_fn_type_id, info.void_type_id});
    }
  }

  // --- Constants ---
  // Emit original constants (already in original_types) — wait, they're in original_types already
  // Actually we already emitted them above in the original_types loop.
  // Only emit the NEW constants:

  // uint32 0, 1, 2 — check if they're new
  {
    bool has0 = false, has1 = false, has2 = false;
    for (auto& [cid, ci] : info.constants) {
      if (ci.type_id == info.uint32_type_id) {
        if (cid == const_uint32_0_id) has0 = true;
        if (cid == const_uint32_1_id) has1 = true;
        if (cid == const_uint32_2_id) has2 = true;
      }
    }
    if (!has0) emitInstr(out, Op::Constant, {info.uint32_type_id, const_uint32_0_id, 0});
    if (!has1) emitInstr(out, Op::Constant, {info.uint32_type_id, const_uint32_1_id, 1});
    if (!has2) emitInstr(out, Op::Constant, {info.uint32_type_id, const_uint32_2_id, 2});
  }

  // New reflection constants (pre-collected above, emit only if not already in original_types)
  for (auto& [value, id] : refl_const_ids) {
    // Skip if already emitted (i.e., was found in info.constants)
    if (info.constants.count(id)) continue;
    emitInstr(out, Op::Constant, {info.uint32_type_id, id, value});
  }

  // WGS spec constants (new)
  emitInstr(out, Op::SpecConstant, {info.uint32_type_id, wgs_x_id, 1});
  emitInstr(out, Op::SpecConstant, {info.uint32_type_id, wgs_y_id, 1});
  emitInstr(out, Op::SpecConstant, {info.uint32_type_id, wgs_z_id, 1});

  // WGS spec constant composite
  emitInstr(out, Op::SpecConstantComposite, {v3uint_type_id, wgs_composite_id, wgs_x_id, wgs_y_id, wgs_z_id});

  // --- Global Variables ---

  // SSBO vars for chip_vars
  for (auto& cv : info.chip_vars) {
    emitInstr(out, Op::Variable, {cv.ssbo_ptr_type_id, cv.ssbo_var_id, SC::StorageBuffer});
  }

  // PC variables for each kernel
  for (auto& [stub_id, kpc] : kernel_pc_info) {
    emitInstr(out, Op::Variable, {kpc.ptr_pc_type_id, kpc.pc_var_id, SC::PushConstant});
  }

  // Also emit any other original global variables (non-chip_var, non-Function)
  // These were already emitted in the original_types loop above

  // =========================================================================
  // Functions
  // =========================================================================

  // Helper: remap an ID using a local remap table
  auto remapId = [](uint32_t id, const std::unordered_map<uint32_t, uint32_t>& remap) -> uint32_t {
    auto it = remap.find(id);
    return it != remap.end() ? it->second : id;
  };

  // For each function in order:
  for (uint32_t fn_id : info.function_order) {
    auto fit = info.functions.find(fn_id);
    if (fit == info.functions.end()) continue;
    FunctionInfo& fn = fit->second;

    bool is_stub = info.stub_fn_ids.count(fn_id) > 0;
    bool is_impl = info.impl_fn_ids.count(fn_id) > 0;

    if (is_stub) {
      // -----------------------------------------------------------------------
      // Emit new kernel stub body
      // -----------------------------------------------------------------------
      auto kpc_it = kernel_pc_info.find(fn_id);
      if (kpc_it == kernel_pc_info.end()) {
        err = "Missing KernelPCInfo for stub fn " + std::to_string(fn_id);
        return {};
      }
      KernelPCInfo& kpc = kpc_it->second;

      uint32_t impl_id = fn.impl_call_target;
      auto iit = info.functions.find(impl_id);
      if (iit == info.functions.end()) {
        err = "Missing impl fn " + std::to_string(impl_id);
        return {};
      }
      FunctionInfo& impl_fn = iit->second;

      // OpFunction %void None %void_fn_type
      emitInstr(out, Op::Function, {info.void_type_id, fn_id, 0, void_fn_type_id});
      // OpLabel
      uint32_t label_id = idAlloc.alloc();
      emitInstr(out, Op::Label, {label_id});

      // For each ByVal struct param: allocate a Function-class local var
      // indexed by arg position
      std::vector<uint32_t> local_struct_var_ids(kpc.layout.size(), 0);
      for (size_t i = 0; i < kpc.layout.size(); ++i) {
        if (kpc.layout[i].is_byval) {
          uint32_t struct_type = kpc.layout[i].type_id;
          auto ptr_fn_it = ptr_function_struct_type_map.find(struct_type);
          if (ptr_fn_it == ptr_function_struct_type_map.end()) {
            err = "Missing ptr Function type for struct " + std::to_string(struct_type);
            return {};
          }
          uint32_t local_var_id = idAlloc.alloc();
          local_struct_var_ids[i] = local_var_id;
          emitInstr(out, Op::Variable, {ptr_fn_it->second, local_var_id, SC::Function});
        }
      }

      // Load each arg from PC and prepare call args
      std::vector<uint32_t> call_args;
      for (size_t i = 0; i < kpc.layout.size(); ++i) {
        ArgLayout& al = kpc.layout[i];
        uint32_t idx_const_id;
        // Get or create index constant for member i
        // Use pre-allocated constants (must NOT emit OpConstant inside function body)
        idx_const_id = getReflConst((uint32_t)i);

        // AccessChain into PC
        uint32_t chain_id = idAlloc.alloc();
        emitInstr(out, Op::AccessChain, {kpc.member_ptr_type_ids[i], chain_id, kpc.pc_var_id, idx_const_id});

        // Load
        uint32_t load_id = idAlloc.alloc();
        uint32_t alignment = typeAlign(al.type_id, info);
        if (alignment == 0) alignment = 4;
        // OpLoad with alignment memory operand: Aligned = 0x2
        emitInstr(out, Op::Load, {al.type_id, load_id, chain_id, 0x2, alignment});

        if (al.is_byval) {
          // Store to local var
          uint32_t local_var = local_struct_var_ids[i];
          emitInstr(out, Op::Store, {local_var, load_id, 0x2, alignment});
          call_args.push_back(local_var);
        } else {
          call_args.push_back(load_id);
        }
      }

      // OpFunctionCall %void %impl_fn_id call_args...
      Words call_words = {info.void_type_id, idAlloc.alloc(), impl_id};
      for (uint32_t arg : call_args) call_words.push_back(arg);
      emitInstr(out, Op::FunctionCall, call_words);

      emitInstr(out, Op::Return, {});
      emitInstr(out, Op::FunctionEnd, {});

    } else if (is_impl) {
      // -----------------------------------------------------------------------
      // Emit transformed impl function
      // -----------------------------------------------------------------------
      uint32_t new_fn_type = impl_fn_new_type_ids.count(fn_id) ? impl_fn_new_type_ids[fn_id] : fn.fn_type;

      // OpFunction
      emitInstr(out, Op::Function, {fn.ret_type, fn_id, 0, new_fn_type});

      // Build ID remap table and emit new params
      std::unordered_map<uint32_t, uint32_t> remap;
      // PSB ptr vars for pointer-kind CW params (to be emitted after OpLabel)
      struct PsbInsert {
        uint32_t ulong_param_id; // the param (now ulong)
        uint32_t psb_result_id;
        uint32_t psb_ptr_type_id;
      };
      std::vector<PsbInsert> psb_inserts;

      auto& np = impl_fn_new_params[fn_id];
      for (size_t i = 0; i < fn.params.size(); ++i) {
        ParamInfo& p = fn.params[i];
        uint32_t new_type = np.new_param_type_ids[i];
        emitInstr(out, Op::FunctionParameter, {new_type, p.id});

        if (p.is_crossworkgroup_ptr) {
          // p.id is now ulong
          if (p.is_pointer_kind) {
            // Need to insert PSB conversion at function entry
            auto pit = info.ptr_types.find(p.type_id);
            if (pit != info.ptr_types.end()) {
              uint32_t base_type = pit->second.base_type;
              auto psb_it = psb_ptr_type_map.find(base_type);
              if (psb_it != psb_ptr_type_map.end()) {
                uint32_t psb_id = idAlloc.alloc();
                psb_inserts.push_back({p.id, psb_id, psb_it->second});
                remap[p.id] = psb_id; // uses of param as ptr → psb_id
              }
            }
          }
          // For pod_kind: remap[p.id] = p.id (it stays ulong, uses in ConvertPtrToU get eliminated)
        }
      }

      // Track result types emitted in this function body, used when handling
      // CopyMemorySized where we need to know the PSB pointee type.
      // Maps result_id → PSB ptr type id (for ConvertUToPtr results).
      std::unordered_map<uint32_t, uint32_t> body_psb_result_type;

      // Emit body instructions (transformed)
      // First, structurize the function body (add OpSelectionMerge/OpLoopMerge).
      const std::vector<Instr> structured_body = structurizeFunctionBody(fn.body, &idAlloc.next_id);
      bool first_label = true;
      for (auto& instr : structured_body) {
        if (instr.opcode == Op::FunctionEnd) {
          emitInstr(out, Op::FunctionEnd, {});
          break;
        }

        if (instr.opcode == Op::Label) {
          emit(out, instr.words);
          if (first_label) {
            first_label = false;
            // Insert PSB conversions
            for (auto& pi : psb_inserts) {
              emitInstr(out, Op::ConvertUToPtr, {pi.psb_ptr_type_id, pi.psb_result_id, pi.ulong_param_id});
            }
          }
          continue;
        }

        // Transform: OpPtrCastToGeneric / OpGenericCastToPtr → no-op in PSB mode
        // These OpenCL address-space casts don't exist in Vulkan; just remap result → source.
        if ((instr.opcode == Op::PtrCastToGeneric || instr.opcode == Op::GenericCastToPtr)
            && instr.wc() >= 4) {
          uint32_t result_id = instr.word(2);
          uint32_t src_id    = remapId(instr.word(3), remap);
          remap[result_id] = src_id;
          continue;
        }

        // Transform: OpConvertPtrToU %ulong %cw_param → remap result to param (pod kind)
        if (instr.opcode == Op::ConvertPtrToU && instr.wc() >= 4) {
          uint32_t result_id  = instr.word(2);
          uint32_t src_id     = remapId(instr.word(3), remap);
          // Check if src was a CW param (now ulong)
          bool src_is_cw_param = false;
          for (auto& p : fn.params) {
            if (p.is_crossworkgroup_ptr && p.id == instr.word(3)) {
              src_is_cw_param = true;
              break;
            }
          }
          if (src_is_cw_param && instr.word(1) == info.uint64_type_id) {
            // Result is just the param itself (already ulong)
            remap[result_id] = instr.word(3); // map to the original param id (now ulong)
            // Don't emit this instruction
            continue;
          }
          // Otherwise emit (with remapped operands)
          Words words = instr.words;
          words[3] = remapId(words[3], remap);
          emit(out, words);
          continue;
        }

        // Transform: OpConvertUToPtr %_ptr_Generic_T → %_ptr_PSB_T
        if (instr.opcode == Op::ConvertUToPtr && instr.wc() >= 4) {
          uint32_t result_type = instr.word(1);
          uint32_t result_id   = instr.word(2);
          uint32_t src         = remapId(instr.word(3), remap);
          auto pit = info.ptr_types.find(result_type);
          if (pit != info.ptr_types.end() &&
              (pit->second.storage_class == SC::Generic ||
               pit->second.storage_class == SC::CrossWorkgroup)) {
            uint32_t base_type = pit->second.base_type;
            auto psb_it = psb_ptr_type_map.find(base_type);
            if (psb_it != psb_ptr_type_map.end()) {
              emitInstr(out, Op::ConvertUToPtr, {psb_it->second, result_id, src});
              body_psb_result_type[result_id] = psb_it->second; // track PSB type for later
              continue;
            }
          }
          // Emit as-is with remapped src
          Words words = instr.words;
          words[3] = src;
          emit(out, words);
          continue;
        }

        // Transform: OpCopyMemorySized with chip_var SSBO source.
        // CopyMemorySized requires Addresses capability (not available in Vulkan).
        // Replace chip_var → PSB copy pattern with AccessChain+Load+Store.
        if (instr.opcode == Op::CopyMemorySized && instr.wc() >= 4) {
          uint32_t dst_id = remapId(instr.word(1), remap);
          uint32_t src_id = instr.word(2); // NOT remapped — check chip_var
          auto cv_it = info.chip_var_by_id.find(src_id);
          if (cv_it != info.chip_var_by_id.end()) {
            ChipVarInfo& cv = info.chip_vars[cv_it->second];
            // Determine PSB pointee type from dst (tracked from ConvertUToPtr)
            // Use chip_var's base type for the load
            uint32_t elem_type = cv.base_type_id;
            uint32_t sb_ptr    = ptr_sb_type_map.count(elem_type)
                                 ? ptr_sb_type_map.at(elem_type) : 0;
            if (sb_ptr && elem_type) {
              // AccessChain into SSBO at index 0 (the single struct member)
              uint32_t chain_id = idAlloc.alloc();
              emitInstr(out, Op::AccessChain, {sb_ptr, chain_id, cv.ssbo_var_id, const_uint32_0_id});
              // Load value from SSBO
              uint32_t val_id = idAlloc.alloc();
              emitInstr(out, Op::Load, {elem_type, val_id, chain_id});
              // Store to PSB destination — must use Aligned for PSB
              uint32_t align = typeAlign(elem_type, info);
              emitInstr(out, Op::Store, {dst_id, val_id, 2 /*Aligned*/, align});
              continue;
            }
          }
          // Case 2: source is a UniformConstant __chip_var_* init variable.
          auto iv_it = info.init_var_constants.find(src_id);
          if (iv_it != info.init_var_constants.end()) {
            // PSB Store requires Aligned; use the natural alignment of the stored type.
            uint32_t const_type = info.constants.count(iv_it->second)
                                  ? info.constants.at(iv_it->second).type_id : 0;
            uint32_t align = const_type ? typeAlign(const_type, info) : 4;
            emitInstr(out, Op::Store, {dst_id, iv_it->second, 2 /*Aligned*/, align});
            continue;
          }

          // Fallback: emit as-is (will likely fail validation)
          Words words = instr.words;
          for (uint32_t i = 1; i < words.size(); ++i)
            words[i] = remapId(words[i], remap);
          emit(out, words);
          continue;
        }

        // Transform: OpInBoundsPtrAccessChain → OpPtrAccessChain with PSB type
        if (instr.opcode == Op::InBoundsPtrAccessChain && instr.wc() >= 5) {
          uint32_t result_type = instr.word(1);
          uint32_t result_id   = instr.word(2);
          uint32_t base        = remapId(instr.word(3), remap);
          auto pit = info.ptr_types.find(result_type);
          if (pit != info.ptr_types.end() &&
              (pit->second.storage_class == SC::Generic ||
               pit->second.storage_class == SC::CrossWorkgroup)) {
            uint32_t base_type = pit->second.base_type;
            auto psb_it = psb_ptr_type_map.find(base_type);
            if (psb_it != psb_ptr_type_map.end()) {
              Words ops = {psb_it->second, result_id, base};
              for (uint32_t i = 4; i < instr.wc(); ++i)
                ops.push_back(remapId(instr.word(i), remap));
              emitInstr(out, Op::PtrAccessChain, ops);
              continue;
            }
          }
          // Function/Private/Workgroup local SC: InBoundsPtrAccessChain is invalid
          // in Vulkan (requires Addresses cap). Convert to PtrAccessChain (Workgroup
          // only) or bail out to clspv fallback (Function/Private SC).
          uint32_t elem_idx_id = instr.word(4);
          auto cit = info.constants.find(elem_idx_id);
          bool elem_is_zero = (cit != info.constants.end() && cit->second.value == 0);
          if (elem_is_zero) {
            // zero offset: result == base pointer
            remap[result_id] = base;
          } else {
            // Only Workgroup SC is valid for PtrAccessChain in Vulkan.
            // Function/Private SC pointer arithmetic cannot be expressed — bail out.
            if (pit == info.ptr_types.end() ||
                pit->second.storage_class != SC::WorkgroupLocal) {
              return {}; // silent fallback to clspv subprocess
            }
            uint32_t elem_idx = remapId(elem_idx_id, remap);
            Words ops = {result_type, result_id, base, elem_idx};
            for (uint32_t i = 5; i < instr.wc(); ++i)
              ops.push_back(remapId(instr.word(i), remap));
            emitInstr(out, Op::PtrAccessChain, ops);
          }
          continue;
        }

        // Transform: chip_var accesses
        // OpLoad %T %chip_var_id → AccessChain into SSBO + load
        if (instr.opcode == Op::Load && instr.wc() >= 4) {
          uint32_t result_type = instr.word(1);
          uint32_t result_id   = instr.word(2);
          uint32_t ptr_id      = remapId(instr.word(3), remap);

          // WorkgroupSize variable: replaced by SpecConstantComposite (wgs_composite_id).
          // Any load from it just reads the spec constant value.
          if (ptr_id == info.workgroup_size_var_id) {
            auto vit = info.vec_types.find(result_type);
            if (vit != info.vec_types.end() &&
                vit->second.first == info.uint64_type_id && vit->second.second == 3) {
              // wgs_composite_id is v3uint; UConvert to v3ulong
              emitInstr(out, Op::UConvert, {result_type, result_id, wgs_composite_id});
            } else {
              // Assume v3uint result (or already matching) — use composite directly
              remap[result_id] = wgs_composite_id;
            }
            continue;
          }

          // BuiltIn Input variable: retype load v3ulong→v3uint, add UConvert to keep
          // downstream result type unchanged (Vulkan requires v3uint for builtins).
          if (info.builtin_input_vars.count(ptr_id)) {
            auto vit = info.vec_types.find(result_type);
            if (vit != info.vec_types.end() &&
                vit->second.first == info.uint64_type_id && vit->second.second == 3) {
              // Load as v3uint (no memory operands — Aligned is OpenCL-specific)
              uint32_t tmp_id = idAlloc.alloc();
              emitInstr(out, Op::Load, {v3uint_type_id, tmp_id, ptr_id});
              // UConvert tmp (v3uint) → result_id (v3ulong)
              emitInstr(out, Op::UConvert, {result_type, result_id, tmp_id});
              continue;
            }
          }

          // Check if ptr_id is a chip_var
          auto cv_it = info.chip_var_by_id.find(ptr_id);
          if (cv_it != info.chip_var_by_id.end()) {
            ChipVarInfo& cv = info.chip_vars[cv_it->second];
            // AccessChain into SSBO
            uint32_t chain_id = idAlloc.alloc();
            uint32_t sb_ptr_type = ptr_sb_type_map.count(cv.base_type_id)
                                   ? ptr_sb_type_map[cv.base_type_id] : 0;
            if (sb_ptr_type == 0) {
              // Create it on the fly (shouldn't happen if pre-scan was complete)
              sb_ptr_type = idAlloc.alloc();
              ptr_sb_type_map[cv.base_type_id] = sb_ptr_type;
              // Can't emit type here in function body... fall through
            }
            emitInstr(out, Op::AccessChain, {sb_ptr_type, chain_id, cv.ssbo_var_id, const_uint32_0_id});
            // Load from chain
            Words load_words = {result_type, result_id, chain_id};
            // Copy memory operands if present
            for (uint32_t i = 4; i < instr.wc(); ++i) load_words.push_back(instr.word(i));
            emitInstr(out, Op::Load, load_words);
            continue;
          }
        }

        // OpStore %chip_var_id %val → AccessChain + store
        if (instr.opcode == Op::Store && instr.wc() >= 3) {
          uint32_t ptr_id = instr.word(1);
          uint32_t val_id = remapId(instr.word(2), remap);
          auto cv_it = info.chip_var_by_id.find(ptr_id);
          if (cv_it != info.chip_var_by_id.end()) {
            ChipVarInfo& cv = info.chip_vars[cv_it->second];
            uint32_t chain_id = idAlloc.alloc();
            uint32_t sb_ptr_type = ptr_sb_type_map.count(cv.base_type_id)
                                   ? ptr_sb_type_map[cv.base_type_id] : 0;
            if (sb_ptr_type != 0) {
              emitInstr(out, Op::AccessChain, {sb_ptr_type, chain_id, cv.ssbo_var_id, const_uint32_0_id});
              Words store_words = {chain_id, val_id};
              for (uint32_t i = 3; i < instr.wc(); ++i) store_words.push_back(instr.word(i));
              emitInstr(out, Op::Store, store_words);
              continue;
            }
          }
        }

        // OpPtrCastToGeneric / OpGenericCastToPtr → no-op in PSB mode
        if ((instr.opcode == Op::PtrCastToGeneric || instr.opcode == Op::GenericCastToPtr)
            && instr.wc() >= 4) {
          uint32_t result_id = instr.word(2);
          uint32_t src_id    = remapId(instr.word(3), remap);
          remap[result_id] = src_id;
          continue;
        }

        // OpConvertPtrToU where ptr is chip_var → AccessChain + load as ulong
        if (instr.opcode == Op::ConvertPtrToU && instr.wc() >= 4) {
          uint32_t result_id = instr.word(2);
          uint32_t ptr_id    = instr.word(3);
          auto cv_it = info.chip_var_by_id.find(ptr_id);
          if (cv_it != info.chip_var_by_id.end()) {
            ChipVarInfo& cv = info.chip_vars[cv_it->second];
            uint32_t chain_id = idAlloc.alloc();
            uint32_t sb_ptr_type = ptr_sb_type_map.count(cv.base_type_id)
                                   ? ptr_sb_type_map[cv.base_type_id] : 0;
            if (sb_ptr_type != 0) {
              emitInstr(out, Op::AccessChain, {sb_ptr_type, chain_id, cv.ssbo_var_id, const_uint32_0_id});
              emitInstr(out, Op::Load, {info.uint64_type_id, result_id, chain_id, 0x2, 8u});
              continue;
            }
          }
        }

        // Protect literal operands in OpLoad/OpStore from ID remapping.
        // MemoryAccess and alignment words are literals, not IDs.
        if (instr.opcode == Op::Store && instr.wc() >= 3) {
          uint32_t ptr = remapId(instr.word(1), remap);
          auto cwit = cw_type_remap.find(ptr);
          if (cwit != cw_type_remap.end()) ptr = cwit->second;
          uint32_t val = remapId(instr.word(2), remap);
          cwit = cw_type_remap.find(val);
          if (cwit != cw_type_remap.end()) val = cwit->second;
          Words ops = {ptr, val};
          for (uint32_t i = 3; i < instr.wc(); ++i) ops.push_back(instr.word(i));
          emitInstr(out, Op::Store, ops);
          continue;
        }
        if (instr.opcode == Op::Load && instr.wc() >= 4) {
          uint32_t type = instr.word(1);
          auto cwit = cw_type_remap.find(type);
          if (cwit != cw_type_remap.end()) type = cwit->second;
          uint32_t result = instr.word(2);
          uint32_t ptr = remapId(instr.word(3), remap);
          cwit = cw_type_remap.find(ptr);
          if (cwit != cw_type_remap.end()) ptr = cwit->second;
          Words ops = {type, result, ptr};
          for (uint32_t i = 4; i < instr.wc(); ++i) ops.push_back(instr.word(i));
          emitInstr(out, Op::Load, ops);
          continue;
        }

        // InBoundsPtrAccessChain is invalid in Vulkan (requires Addresses cap).
        // Convert to PtrAccessChain for valid SC, or remap for zero element index.
        if (instr.opcode == Op::InBoundsPtrAccessChain && instr.wc() >= 5) {
          uint32_t orig_type   = instr.word(1);
          uint32_t result_type = orig_type;
          uint32_t result_id   = instr.word(2);
          uint32_t base        = remapId(instr.word(3), remap);
          // Apply cw_type_remap to result type (CW/Generic → PSB)
          auto cwit = cw_type_remap.find(result_type);
          bool was_remapped = (cwit != cw_type_remap.end());
          if (was_remapped) result_type = cwit->second;
          uint32_t elem_idx_id = instr.word(4);
          auto cit = info.constants.find(elem_idx_id);
          bool elem_is_zero = (cit != info.constants.end() && cit->second.value == 0);
          if (elem_is_zero) {
            remap[result_id] = base;
          } else {
            // PtrAccessChain is only valid for PSB, Workgroup, StorageBuffer SC.
            // If not remapped to PSB and not Workgroup/StorageBuffer, bail out.
            if (!was_remapped) {
              auto orig_pit = info.ptr_types.find(orig_type);
              if (orig_pit == info.ptr_types.end() ||
                  (orig_pit->second.storage_class != SC::WorkgroupLocal &&
                   orig_pit->second.storage_class != SC::StorageBuffer)) {
                return {}; // silent fallback to clspv subprocess
              }
            }
            uint32_t elem_idx = remapId(elem_idx_id, remap);
            Words ops = {result_type, result_id, base, elem_idx};
            for (uint32_t i = 5; i < instr.wc(); ++i)
              ops.push_back(remapId(instr.word(i), remap));
            emitInstr(out, Op::PtrAccessChain, ops);
          }
          continue;
        }

        // Default: emit instruction with remapped IDs
        Words words = instr.words;
        // Remap all operand words: apply both value remap and CW→PSB type remap.
        // (false positives are impossible since both maps only contain known IDs)
        for (uint32_t i = 1; i < words.size(); ++i) {
          uint32_t w = remapId(words[i], remap);
          // Also replace any CW/Generic pointer type IDs with PSB pointer type IDs
          auto cwit = cw_type_remap.find(w);
          if (cwit != cw_type_remap.end()) w = cwit->second;
          words[i] = w;
        }
        emit(out, words);
      }

    } else {
      // -----------------------------------------------------------------------
      // Non-stub, non-impl function: emit with chip_var and PSB transforms.
      // These include __chip_var_init_A, __chip_var_bind_A, etc. which use
      // OpCopyMemorySized, ConvertUToPtr with CW types, and chip_var loads.
      // -----------------------------------------------------------------------
      {
        uint32_t eff_fn_type = fn.fn_type;
        auto hnit = helper_fn_new_type_ids.find(fn_id);
        if (hnit != helper_fn_new_type_ids.end())
          eff_fn_type = hnit->second;
        emit(out, makeInstr(Op::Function, {fn.ret_type, fn_id, 0, eff_fn_type}));
      }
      {
        auto hnpit = helper_fn_new_params.find(fn_id);
        for (size_t pi = 0; pi < fn.params.size(); ++pi) {
          auto& p = fn.params[pi];
          uint32_t eff_type = p.type_id;
          if (hnpit != helper_fn_new_params.end() &&
              pi < hnpit->second.new_param_type_ids.size())
            eff_type = hnpit->second.new_param_type_ids[pi];
          emitInstr(out, Op::FunctionParameter, {eff_type, p.id});
        }
      }
      // Empty remap (no parameter remapping for these functions)
      std::unordered_map<uint32_t, uint32_t> remap;
      std::unordered_map<uint32_t, uint32_t> body_psb_result_type;

      const std::vector<Instr> structured_body2 = structurizeFunctionBody(fn.body, &idAlloc.next_id);
      for (auto& instr : structured_body2) {
        if (instr.opcode == Op::FunctionEnd) {
          emitInstr(out, Op::FunctionEnd, {});
          break;
        }

        // Transform: OpConvertUToPtr %_ptr_Generic/CW_T → %_ptr_PSB_T
        if (instr.opcode == Op::ConvertUToPtr && instr.wc() >= 4) {
          uint32_t result_type = instr.word(1);
          uint32_t result_id   = instr.word(2);
          uint32_t src         = remapId(instr.word(3), remap);
          auto pit = info.ptr_types.find(result_type);
          if (pit != info.ptr_types.end() &&
              (pit->second.storage_class == SC::Generic ||
               pit->second.storage_class == SC::CrossWorkgroup)) {
            uint32_t base_type = pit->second.base_type;
            auto psb_it = psb_ptr_type_map.find(base_type);
            if (psb_it != psb_ptr_type_map.end()) {
              emitInstr(out, Op::ConvertUToPtr, {psb_it->second, result_id, src});
              body_psb_result_type[result_id] = psb_it->second;
              continue;
            }
          }
          Words words = instr.words;
          words[3] = src;
          emit(out, words);
          continue;
        }

        // Transform: OpCopyMemorySized with chip_var SSBO source or init_var constant source.
        if (instr.opcode == Op::CopyMemorySized && instr.wc() >= 4) {
          uint32_t dst_id = remapId(instr.word(1), remap);
          uint32_t src_id = instr.word(2);

          // Case 1: source is a chip_var SSBO (CrossWorkgroup)
          auto cv_it = info.chip_var_by_id.find(src_id);
          if (cv_it != info.chip_var_by_id.end()) {
            ChipVarInfo& cv = info.chip_vars[cv_it->second];
            uint32_t elem_type = cv.base_type_id;
            uint32_t sb_ptr    = ptr_sb_type_map.count(elem_type)
                                 ? ptr_sb_type_map.at(elem_type) : 0;
            if (sb_ptr && elem_type) {
              uint32_t chain_id = idAlloc.alloc();
              emitInstr(out, Op::AccessChain, {sb_ptr, chain_id, cv.ssbo_var_id, const_uint32_0_id});
              uint32_t val_id = idAlloc.alloc();
              emitInstr(out, Op::Load, {elem_type, val_id, chain_id});
              uint32_t align = typeAlign(elem_type, info);
              emitInstr(out, Op::Store, {dst_id, val_id, 2 /*Aligned*/, align});
              continue;
            }
          }

          // Case 2: source is a UniformConstant __chip_var_*_initializer variable.
          // Replace CopyMemorySized with a direct Store of the embedded constant.
          auto iv_it = info.init_var_constants.find(src_id);
          if (iv_it != info.init_var_constants.end()) {
            // PSB Store requires Aligned; use the natural alignment of the stored type.
            uint32_t const_type = info.constants.count(iv_it->second)
                                  ? info.constants.at(iv_it->second).type_id : 0;
            uint32_t align = const_type ? typeAlign(const_type, info) : 4;
            emitInstr(out, Op::Store, {dst_id, iv_it->second, 2 /*Aligned*/, align});
            continue;
          }

          // Fallback: emit as-is (will likely fail validation)
          Words words = instr.words;
          for (uint32_t i = 1; i < words.size(); ++i)
            words[i] = remapId(words[i], remap);
          emit(out, words);
          continue;
        }

        // Transform: chip_var Load
        if (instr.opcode == Op::Load && instr.wc() >= 4) {
          uint32_t result_type = instr.word(1);
          uint32_t result_id   = instr.word(2);
          uint32_t ptr_id      = instr.word(3);

          // WorkgroupSize: replaced by SpecConstantComposite
          if (ptr_id == info.workgroup_size_var_id) {
            auto vit = info.vec_types.find(result_type);
            if (vit != info.vec_types.end() &&
                vit->second.first == info.uint64_type_id && vit->second.second == 3) {
              emitInstr(out, Op::UConvert, {result_type, result_id, wgs_composite_id});
            } else {
              remap[result_id] = wgs_composite_id;
            }
            continue;
          }

          // BuiltIn Input variable: retype load v3ulong→v3uint + UConvert
          if (info.builtin_input_vars.count(ptr_id)) {
            auto vit = info.vec_types.find(result_type);
            if (vit != info.vec_types.end() &&
                vit->second.first == info.uint64_type_id && vit->second.second == 3) {
              uint32_t tmp_id = idAlloc.alloc();
              emitInstr(out, Op::Load, {v3uint_type_id, tmp_id, ptr_id});
              emitInstr(out, Op::UConvert, {result_type, result_id, tmp_id});
              continue;
            }
          }

          auto cv_it = info.chip_var_by_id.find(ptr_id);
          if (cv_it != info.chip_var_by_id.end()) {
            ChipVarInfo& cv = info.chip_vars[cv_it->second];
            uint32_t sb_ptr_type = ptr_sb_type_map.count(cv.base_type_id)
                                   ? ptr_sb_type_map[cv.base_type_id] : 0;
            if (sb_ptr_type != 0) {
              uint32_t chain_id = idAlloc.alloc();
              emitInstr(out, Op::AccessChain, {sb_ptr_type, chain_id, cv.ssbo_var_id, const_uint32_0_id});
              Words load_words = {result_type, result_id, chain_id};
              for (uint32_t i = 4; i < instr.wc(); ++i) load_words.push_back(instr.word(i));
              emitInstr(out, Op::Load, load_words);
              continue;
            }
          }
        }

        // Transform: chip_var Store
        if (instr.opcode == Op::Store && instr.wc() >= 3) {
          uint32_t ptr_id = instr.word(1);
          uint32_t val_id = remapId(instr.word(2), remap);
          auto cv_it = info.chip_var_by_id.find(ptr_id);
          if (cv_it != info.chip_var_by_id.end()) {
            ChipVarInfo& cv = info.chip_vars[cv_it->second];
            uint32_t sb_ptr_type = ptr_sb_type_map.count(cv.base_type_id)
                                   ? ptr_sb_type_map[cv.base_type_id] : 0;
            if (sb_ptr_type != 0) {
              uint32_t chain_id = idAlloc.alloc();
              emitInstr(out, Op::AccessChain, {sb_ptr_type, chain_id, cv.ssbo_var_id, const_uint32_0_id});
              Words store_words = {chain_id, val_id};
              for (uint32_t i = 3; i < instr.wc(); ++i) store_words.push_back(instr.word(i));
              emitInstr(out, Op::Store, store_words);
              continue;
            }
          }
        }

        // Protect literal operands in OpLoad/OpStore from ID remapping.
        if (instr.opcode == Op::Store && instr.wc() >= 3) {
          uint32_t ptr = remapId(instr.word(1), remap);
          auto cwit = cw_type_remap.find(ptr);
          if (cwit != cw_type_remap.end()) ptr = cwit->second;
          uint32_t val = remapId(instr.word(2), remap);
          cwit = cw_type_remap.find(val);
          if (cwit != cw_type_remap.end()) val = cwit->second;
          Words ops = {ptr, val};
          for (uint32_t i = 3; i < instr.wc(); ++i) ops.push_back(instr.word(i));
          emitInstr(out, Op::Store, ops);
          continue;
        }
        if (instr.opcode == Op::Load && instr.wc() >= 4) {
          uint32_t type = instr.word(1);
          auto cwit = cw_type_remap.find(type);
          if (cwit != cw_type_remap.end()) type = cwit->second;
          uint32_t result = instr.word(2);
          uint32_t ptr = remapId(instr.word(3), remap);
          cwit = cw_type_remap.find(ptr);
          if (cwit != cw_type_remap.end()) ptr = cwit->second;
          Words ops = {type, result, ptr};
          for (uint32_t i = 4; i < instr.wc(); ++i) ops.push_back(instr.word(i));
          emitInstr(out, Op::Load, ops);
          continue;
        }

        // Default: emit with remap + CW→PSB type remap
        Words words = instr.words;
        for (uint32_t i = 1; i < words.size(); ++i) {
          uint32_t w = remapId(words[i], remap);
          auto cwit = cw_type_remap.find(w);
          if (cwit != cw_type_remap.end()) w = cwit->second;
          words[i] = w;
        }
        emit(out, words);
      }
    }
  }

  // =========================================================================
  // NonSemantic.ClspvReflection metadata (after all functions)
  // =========================================================================

  // All OpString IDs were pre-allocated and emitted in the debug section.
  // Use allocStringId as a lookup function here (no new emission).
  auto getStringId = [&](const std::string& s) -> uint32_t {
    return allocStringId(s); // IDs already allocated; new calls just look up
  };

  // OpExtInst returns void type; all reflection uses info.void_type_id.

  for (auto& ep : info.entry_points) {
    uint32_t stub_id = ep.fn_id;
    auto kpc_it = kernel_pc_info.find(stub_id);
    if (kpc_it == kernel_pc_info.end()) continue;
    KernelPCInfo& kpc = kpc_it->second;

    uint32_t impl_id = 0;
    {
      auto sit = info.functions.find(stub_id);
      if (sit != info.functions.end())
        impl_id = sit->second.impl_call_target;
    }
    if (impl_id == 0) continue;
    auto iit = info.functions.find(impl_id);
    if (iit == info.functions.end()) continue;
    FunctionInfo& impl_fn = iit->second;

    uint32_t kernel_name_str = getStringId(ep.name);
    uint32_t attr_str        = getStringId(""); // empty kernel attributes

    // Use pre-allocated reflection constants (emitted in constants section, not here)
    uint32_t specid0_const = getReflConst(0);
    uint32_t specid1_const = getReflConst(1);
    uint32_t specid2_const = getReflConst(2);
    uint32_t num_args_const = getReflConst((uint32_t)kpc.layout.size());

    // Kernel reflection: Kernel(fn_id, name_str, num_args_const, wgs_x_specid, attr_str)
    uint32_t kernel_refl_id = idAlloc.alloc();
    emitInstr(out, Op::ExtInst, {
      info.void_type_id, kernel_refl_id, clspv_refl_id,
      NSRefl::Kernel, stub_id, kernel_name_str,
      num_args_const, specid0_const, attr_str
    });

    // WorkgroupSize spec constant info: SpecConstantWorkgroupSize(specid_x, specid_y, specid_z)
    // arguments are the SpecId VALUE integers (0, 1, 2), not the variable IDs
    uint32_t wgs_refl_id = idAlloc.alloc();
    emitInstr(out, Op::ExtInst, {
      info.void_type_id, wgs_refl_id, clspv_refl_id,
      NSRefl::SpecConstantWorkgroupSize,
      specid0_const, specid1_const, specid2_const
    });

    // Argument reflection
    for (size_t i = 0; i < kpc.layout.size(); ++i) {
      ArgLayout& al = kpc.layout[i];
      // Arg name (try to get from param name or use "")
      std::string arg_name = "";
      if (i < impl_fn.params.size()) {
        auto nit = info.names.find(impl_fn.params[i].id);
        if (nit != info.names.end()) arg_name = nit->second;
      }
      uint32_t arg_name_str = getStringId(arg_name);

      // ArgumentInfo(arg_name_str) → arg_info_id
      uint32_t arg_info_id = idAlloc.alloc();
      emitInstr(out, Op::ExtInst, {
        info.void_type_id, arg_info_id, clspv_refl_id,
        NSRefl::ArgumentInfo, arg_name_str
      });

      uint32_t kind = al.is_ulong ? NSRefl::ArgumentPointerPushConstant
                                   : NSRefl::ArgumentPodPushConstant;
      uint32_t arg_ordinal_const = getReflConst((uint32_t)i);
      uint32_t pc_offset_const   = getReflConst(al.pc_offset);
      uint32_t pc_size_const     = getReflConst(al.pc_size);
      // Arg reflection: kind(kernel_id, ordinal, offset, size, arg_info_id)
      uint32_t arg_refl_id = idAlloc.alloc();
      emitInstr(out, Op::ExtInst, {
        info.void_type_id, arg_refl_id, clspv_refl_id,
        kind, kernel_refl_id, arg_ordinal_const,
        pc_offset_const, pc_size_const, arg_info_id
      });
    }

  }  // end per-kernel loop

  // ProgramScopeVariables reflection for chip_vars (module-level, not per-kernel)
  for (size_t ci = 0; ci < info.chip_vars.size(); ++ci) {
    ChipVarInfo& cv = info.chip_vars[ci];

    // Determine the byte size of the base type to generate hex-encoded zeros.
    uint32_t type_bytes = 8; // default: 8 bytes (ulong)
    {
      auto wit = info.int_widths.find(cv.base_type_id);
      if (wit != info.int_widths.end())
        type_bytes = (wit->second + 7) / 8;
      else {
        auto fit = info.float_widths.find(cv.base_type_id);
        if (fit != info.float_widths.end())
          type_bytes = (fit->second + 7) / 8;
      }
    }
    // Hex string of zero bytes (chip_vars are externally initialized at runtime)
    std::string hex_data(type_bytes * 2, '0');
    uint32_t data_str_id = getStringId(hex_data);

    uint32_t ds_const  = getReflConst(cv.descriptor_set);
    uint32_t bd_const  = getReflConst(cv.binding);
    uint32_t psv_id = idAlloc.alloc();
    emitInstr(out, Op::ExtInst, {
      info.void_type_id, psv_id, clspv_refl_id,
      NSRefl::ProgramScopeVariablesStorageBuffer,
      ds_const, bd_const, data_str_id
    });
  }

  // Update bound in header (word index 3)
  out[3] = idAlloc.next_id;

  return out;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<uint32_t> openclToVulkanSpirv(const std::vector<uint8_t>& opencl_spv,
                                           const std::string& clspv_options,
                                           std::string* error_msg) {
  auto SetErr = [&](const std::string& Msg) -> std::vector<uint32_t> {
    logError("openclToVulkanSpirv: {}", Msg);
    if (error_msg) *error_msg = Msg;
    return {};
  };

  // Only handle PSB mode
  bool HasPSB = clspv_options.find("-physical-storage-buffers") != std::string::npos;
  if (!HasPSB) {
    // Return empty — caller should fall back to clvk's pipeline
    logDebug("openclToVulkanSpirv: no -physical-storage-buffers, returning empty (use clvk pipeline)");
    return {};
  }

  // Validate input
  if (opencl_spv.empty()) return SetErr("Empty input SPIR-V");
  if (opencl_spv.size() % 4 != 0) return SetErr("Input SPIR-V size not a multiple of 4");

  // Convert byte array to word array
  Words spv(opencl_spv.size() / 4);
  std::memcpy(spv.data(), opencl_spv.data(), opencl_spv.size());

  if (spv.size() < 5 || spv[0] != spv::MagicNumber) {
    return SetErr("Invalid SPIR-V magic number");
  }

  // Pass 1: collect module info
  ModuleInfo info;
  std::string scan_err;
  if (!collectModuleInfo(spv, info, scan_err)) {
    return SetErr("Pass 1 failed: " + scan_err);
  }

  // Check if we have entry points
  if (info.entry_points.empty()) {
    return SetErr("No entry points found in SPIR-V");
  }

  // Pass 2: emit transformed SPIR-V
  std::string emit_err;
  Words result = emitVulkanSpirv(info, emit_err);
  if (result.empty()) {
    if (!emit_err.empty())
      return SetErr("Pass 2 failed: " + emit_err);
    // Empty result from unsupported pattern (legitimate fallback)
    logDebug("openclToVulkanSpirv: unsupported pattern, returning empty (use clvk pipeline)");
    return {};
  }

  logDebug("openclToVulkanSpirv: produced {} words of Vulkan PSB SPIR-V", result.size());

  // Debug dump: write to /tmp/spirv_to_vulkan_debug.spv if CHIP_DUMP_VK_SPIRV is set
  if (getenv("CHIP_DUMP_VK_SPIRV")) {
    std::string path(getenv("CHIP_DUMP_VK_SPIRV"));
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(result.data()), result.size() * 4);
    logInfo("openclToVulkanSpirv: dumped {} words to {}", result.size(), path);
  }
  if (getenv("CHIP_DUMP_OCL_SPIRV")) {
    std::string path(getenv("CHIP_DUMP_OCL_SPIRV"));
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(spv.data()), spv.size() * 4);
  }

  return result;
}

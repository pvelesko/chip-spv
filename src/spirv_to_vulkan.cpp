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
#include <cstring>
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
  static const uint32_t ISub                    = 130;
  static const uint32_t IAdd                    = 128;
  static const uint32_t BitCount                = 205;
  static const uint32_t IEqual                  = 170;
  static const uint32_t Select                  = 169;
  static const uint32_t ShiftRightLogical       = 194;
  static const uint32_t TypeBool                = 20;
  static const uint32_t IMul                    = 132;
  static const uint32_t BitwiseAnd              = 199;
  static const uint32_t Label                   = 248;
  static const uint32_t Branch                  = 249;
  static const uint32_t BranchConditional       = 250;
  static const uint32_t Switch                  = 251;
  static const uint32_t CompositeConstruct        = 80;
  static const uint32_t CompositeExtract         = 81;
  static const uint32_t UConvert                 = 113;
  static const uint32_t INotEqual               = 171;
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
  static const uint32_t SpecId            = 1;
  static const uint32_t Block             = 2;
  static const uint32_t BuiltIn           = 11;
  static const uint32_t Binding           = 33;
  static const uint32_t DescriptorSet     = 34;
  static const uint32_t Offset            = 35;
  static const uint32_t FuncParamAttr     = 38;
  static const uint32_t Alignment         = 44;
  static const uint32_t ArrayStride       = 6;
  static const uint32_t Constant          = 22; // OpenCL-only: read-only marker
  static const uint32_t LinkageAttributes = 41; // OpenCL-only: cross-module linkage
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
  // ExtInstImport ID -> set name (e.g. 1 -> "OpenCL.std")
  std::unordered_map<uint32_t, std::string> ext_inst_import_names;
  // The id of the "OpenCL.std" ExtInstImport, if present (0 if not).
  uint32_t opencl_std_import_id = 0;

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
  uint32_t bool_type_id   = 0; // OpTypeBool — cannot be stored in push constants

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
      if (instr.wc() >= 2) {
        uint32_t import_id = instr.word(1);
        info.ext_inst_import_ids.insert(import_id);
        std::string name = decodeLiteralString(instr.words, 2);
        info.ext_inst_import_names[import_id] = name;
        if (name == "OpenCL.std")
          info.opencl_std_import_id = import_id;
      }
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
      if (instr.wc() >= 2 && !in_function) {
        info.bool_type_id = instr.word(1);
        info.original_types.push_back(instr);
      }
      break;
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
          // Treat every CrossWorkgroup global as an SSBO-backed shared
          // variable. The chip_var pipeline replaces it with an
          // SSBO descriptor + struct wrapper. This handles:
          //   __chip_var_*                — wrapped __device__ symbols
          //   __chip_module_has_no_IGBAs  — IGBA marker
          //   __chipspv_*                 — runtime helpers (device_heap, etc.)
          //   __chip_clk_counter          — clock() intrinsic counter
          //   user __device__ constants   — exported via LinkageAttributes
          // For export-only constants the chipStar runtime won't bind a
          // buffer, but that's fine if no kernel actually reads them; the
          // init/bind helpers just write to an unbound SSBO and the SPIR-V
          // remains valid.
          ChipVarInfo cv;
          cv.id           = id;
          auto nit = info.names.find(id);
          cv.name         = (nit != info.names.end())
                              ? nit->second
                              : ("__chip_anon_" + std::to_string(id));
          cv.ptr_type_id  = ptr_type;
          auto pit = info.ptr_types.find(ptr_type);
          if (pit != info.ptr_types.end())
            cv.base_type_id = pit->second.base_type;
          info.chip_var_by_id[id] = info.chip_vars.size();
          info.chip_vars.push_back(cv);
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
        if (pit->second.storage_class == SC::WorkgroupLocal)
          // Dynamic shared memory param — cannot be expressed as push constant or
          // StorageBuffer.  Fall back to clvk's internal clspv pipeline.
          info.has_unsupported_globals = true;
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
  auto classifyImplParams = [](FunctionInfo& fn) {
    std::unordered_set<uint32_t> cw_params;
    for (auto& p : fn.params)
      if (p.is_crossworkgroup_ptr) cw_params.insert(p.id);
    if (cw_params.empty()) return;

    std::unordered_set<uint32_t> pod_params;
    std::unordered_set<uint32_t> ptr_params;
    for (auto& instr : fn.body) {
      if (instr.opcode == Op::ConvertPtrToU && instr.wc() >= 4) {
        uint32_t src = instr.word(3);
        if (cw_params.count(src)) pod_params.insert(src);
      }
      if ((instr.opcode == Op::Load || instr.opcode == Op::Store ||
           instr.opcode == Op::AccessChain || instr.opcode == Op::InBoundsAccessChain ||
           instr.opcode == Op::PtrAccessChain || instr.opcode == Op::InBoundsPtrAccessChain) &&
          instr.wc() >= 3) {
        uint32_t base_idx = (instr.opcode == Op::Store) ? 1 : 3;
        if (base_idx < instr.wc()) {
          uint32_t base = instr.word(base_idx);
          if (cw_params.count(base)) ptr_params.insert(base);
        }
      }
      if (instr.opcode == Op::FunctionCall && instr.wc() >= 4) {
        for (uint32_t ai = 3; ai < instr.wc(); ++ai)
          if (cw_params.count(instr.word(ai))) ptr_params.insert(instr.word(ai));
      }
    }
    for (auto& p : fn.params) {
      if (!p.is_crossworkgroup_ptr) continue;
      bool pod_use = pod_params.count(p.id) > 0;
      bool ptr_use = ptr_params.count(p.id) > 0;
      if (!pod_use && !ptr_use) p.is_pod_kind = true;
      else if (ptr_use) p.is_pointer_kind = true;
      else p.is_pod_kind = true;
    }
  };

  for (auto& fn_id : info.impl_fn_ids) {
    auto it = info.functions.find(fn_id);
    if (it == info.functions.end()) continue;
    classifyImplParams(it->second);
  }

  // Synthesize stub-impl pairs for inline kernels (entry-point functions whose
  // bodies are not just a single FunctionCall to an impl). Without this, the
  // entry-point function keeps its OpenCL kernel parameters, but Vulkan SPIR-V
  // requires entry points to have zero parameters (VUID-StandaloneSpirv-None-04633).
  // We synthesize a fresh impl id, move the body+params there, and let the stub
  // emit branch later rebuild a zero-parameter wrapper that loads args from
  // push constants and calls the new impl.
  //
  // Note: new impl ids must not collide with the SPIR-V Bound. We use ids
  // starting at info.bound and bump info.bound accordingly so subsequent
  // idAlloc uses see the right base.
  {
    std::unordered_set<uint32_t> ep_fn_ids;
    for (auto& ep : info.entry_points) ep_fn_ids.insert(ep.fn_id);
    std::vector<uint32_t> inline_kernel_ids;
    for (uint32_t ep_id : ep_fn_ids) {
      if (info.stub_fn_ids.count(ep_id)) continue;
      auto fit = info.functions.find(ep_id);
      if (fit == info.functions.end()) continue;
      // Skip if the entry point has no parameters (already Vulkan-compatible).
      if (fit->second.params.empty()) continue;
      inline_kernel_ids.push_back(ep_id);
    }
    for (uint32_t orig_id : inline_kernel_ids) {
      uint32_t new_impl_id = info.bound++;
      // Take a value copy first; the unordered_map insert below may rehash and
      // invalidate references obtained from operator[].
      FunctionInfo new_impl = info.functions[orig_id];
      info.functions.emplace(new_impl_id, std::move(new_impl));
      FunctionInfo& orig = info.functions[orig_id];
      orig.is_kernel_stub = true;
      orig.impl_call_target = new_impl_id;
      orig.body.clear();
      orig.params.clear();
      info.stub_fn_ids.insert(orig_id);
      info.impl_fn_ids.insert(new_impl_id);
      // Insert the new impl in function_order right after the original entry
      // point so the SPIR-V module structure stays well-defined (impl follows
      // its caller stub in the binary).
      auto fo_it = std::find(info.function_order.begin(),
                             info.function_order.end(), orig_id);
      if (fo_it == info.function_order.end())
        info.function_order.push_back(new_impl_id);
      else
        info.function_order.insert(fo_it + 1, new_impl_id);
      classifyImplParams(info.functions[new_impl_id]);
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
  // For byval struct types that are also used as Function-scope variable base
  // types: SPIR-V VUID-Offset-04547 forbids Offset decorations on structs used
  // in Function/Private storage class, but our PushConstant Block-decorated
  // struct needs Offsets on its members (recursively). Resolution: clone the
  // struct type. clone_type_id is the new id used in PC layout (Block context);
  // the original (type_id) stays for the Function-scope local variable. The
  // stub emits a member-by-member CompositeExtract+CompositeConstruct to
  // convert from clone-typed to original-typed value before the local Store.
  uint32_t clone_type_id = 0;
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
      type_id = param.type_id;
      // bool cannot be stored in push constants: use uint32 as the storage type.
      if (type_id == info.bool_type_id)
        type_id = info.uint32_type_id;
      sz    = typeSize(type_id, info);
      align = typeAlign(type_id, info);
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
  // Track which blocks are already declared as merge blocks so we can avoid
  // declaring a block as the merge target for two different headers — SPIR-V
  // requires each block to be a merge block for at most one header.
  std::vector<Instr> result;
  std::vector<uint32_t> synthetic_merges; // IDs of synthetic unreachable merge blocks
  // fwd_merges[fwd_id] = original_merge_id: synthetic forwarding blocks that
  // branch to original_merge_id (inserted when original_merge_id is already used).
  std::vector<std::pair<uint32_t,uint32_t>> fwd_merges;
  std::unordered_set<uint32_t> used_merge_set; // blocks already declared as merge targets

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
            used_merge_set.insert(loop_merge_label[blk.label_id]);
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
              // If the merge block is already used by another header, insert a
              // synthetic forwarding block M' → merge instead of using merge directly.
              // SPIR-V requires: a block can be a merge block for at most one header.
              if (used_merge_set.count(merge) && next_id) {
                uint32_t fwd = (*next_id)++;
                fwd_merges.push_back({fwd, merge});
                // Modify the BranchConditional to replace 'merge' with 'fwd'.
                // We'll emit a modified instruction below.
                result.push_back(makeSynthInstr(Op::SelectionMerge, {fwd, 0}));
                // Build modified BranchConditional: replace merge → fwd in targets.
                Instr mod = ins;
                if (mod.words.size() > 2 && mod.words[2] == merge) mod.words[2] = fwd;
                if (mod.words.size() > 3 && mod.words[3] == merge) mod.words[3] = fwd;
                // Recompute header word (wc unchanged)
                result.push_back(mod);
                used_merge_set.insert(fwd);
                continue; // skip the default 'result.push_back(ins)' below
              }
              result.push_back(makeSynthInstr(Op::SelectionMerge, {merge, 0}));
              used_merge_set.insert(merge);
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
              if (merge != 0) {
                if (used_merge_set.count(merge) && next_id) {
                  uint32_t fwd = (*next_id)++;
                  fwd_merges.push_back({fwd, merge});
                  result.push_back(makeSynthInstr(Op::SelectionMerge, {fwd, 0}));
                  used_merge_set.insert(fwd);
                  // Modify switch targets — only default matters for structured merge
                  // (case labels are emitted as-is; they'll exit the construct correctly).
                  // This is a best-effort fix for the common case.
                } else {
                  result.push_back(makeSynthInstr(Op::SelectionMerge, {merge, 0}));
                  used_merge_set.insert(merge);
                }
              }
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
  // Append forwarding merge blocks: fwd_id → original_merge_id.
  for (auto& [fwd, orig] : fwd_merges) {
    result.push_back(makeSynthInstr(Op::Label, {fwd}));
    result.push_back(makeSynthInstr(Op::Branch, {orig}));
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
// Post-processing: fix OpStore/OpLoad alignment for PSB pointers.
// VUID-StandaloneSpirv-PhysicalStorageBuffer64-06314: the Aligned operand
// must be >= the ABI alignment of the largest scalar type in the pointee.
// We copy alignments verbatim from the OpenCL SPIR-V where the PSB rule
// doesn't apply; this pass corrects them in the Vulkan output.
// ---------------------------------------------------------------------------

static void fixPSBAlignments(Words& spv, const ModuleInfo& info) {
  // Step 0: collect all PhysicalStorageBuffer pointer type IDs and their base
  // types from the binary. We cannot rely solely on info.ptr_types because new
  // PSB types allocated by idAlloc during emitVulkanSpirv are not in info.ptr_types.
  std::unordered_set<uint32_t> psb_ptr_type_ids;
  std::unordered_map<uint32_t, uint32_t> psb_ptr_base; // psb_ptr_type_id → base_type_id
  {
    // Seed from info.ptr_types (original OCL types, in case any remain)
    for (auto& [tid, pti] : info.ptr_types) {
      if (pti.storage_class == SC::PhysicalStorageBuffer) {
        psb_ptr_type_ids.insert(tid);
        psb_ptr_base[tid] = pti.base_type;
      }
    }
    // Scan the binary for OpTypePointer PhysicalStorageBuffer
    size_t pos = 5, n = spv.size();
    while (pos < n) {
      uint32_t wc = spv[pos] >> 16;
      uint32_t op = spv[pos] & 0xFFFF;
      if (wc == 0 || pos + wc > n) break;
      if (op == 32 /*OpTypePointer*/ && wc >= 4) {
        if (spv[pos+2] == SC::PhysicalStorageBuffer) {
          psb_ptr_type_ids.insert(spv[pos+1]);
          psb_ptr_base[spv[pos+1]] = spv[pos+3]; // base_type_id
        }
      }
      pos += wc;
    }
  }

  // Step 1: scan SPIR-V to build result_id → ptr_type_id for all instructions
  // that produce a PhysicalStorageBuffer pointer.
  std::unordered_map<uint32_t, uint32_t> id_psb_type; // result_id → psb ptr type
  {
    size_t pos = 5, n = spv.size();
    while (pos < n) {
      uint32_t wc = spv[pos] >> 16;
      uint32_t op = spv[pos] & 0xFFFF;
      if (wc == 0 || pos + wc > n) break;
      // Instructions that produce a typed result with result at word[2]:
      // FunctionParameter(55), ConvertUToPtr(70), Bitcast(124),
      // AccessChain(65), InBoundsAccessChain(66),
      // PtrAccessChain(77), InBoundsPtrAccessChain(78), Load(61)
      if (wc >= 3) {
        uint32_t type_id   = spv[pos+1];
        uint32_t result_id = spv[pos+2];
        // FunctionParameter (55): wc = 3
        if (op == 55 /*FunctionParameter*/) {
          if (psb_ptr_type_ids.count(type_id))
            id_psb_type[result_id] = type_id;
        }
        // Other typed result instructions: wc >= 4
        // Opcodes: Load=61, AccessChain=65, InBoundsAccessChain=66,
        //          PtrAccessChain=67, InBoundsPtrAccessChain=70,
        //          ConvertUToPtr=120, Bitcast=124
        if (wc >= 4) {
          static const uint32_t typed_result_ops[] = {
            61 /*Load*/, 65 /*AccessChain*/, 66 /*InBoundsAccessChain*/,
            67 /*PtrAccessChain*/, 70 /*InBoundsPtrAccessChain*/,
            120 /*ConvertUToPtr*/, 124 /*Bitcast*/
          };
          for (uint32_t tracked_op : typed_result_ops) {
            if (op == tracked_op) {
              if (psb_ptr_type_ids.count(type_id))
                id_psb_type[result_id] = type_id;
              break;
            }
          }
        }
      }
      pos += wc;
    }
  }
  if (id_psb_type.empty()) return; // nothing to fix

  // Step 2: scan for OpStore/OpLoad with Aligned and fix if pointer is PSB.
  size_t pos = 5, n = spv.size();
  while (pos < n) {
    uint32_t wc = spv[pos] >> 16;
    uint32_t op = spv[pos] & 0xFFFF;
    if (wc == 0 || pos + wc > n) break;

    // OpStore: ptr=word[1], val=word[2], [mem_mask=word[3], align=word[4]]
    if (op == Op::Store && wc >= 5) {
      uint32_t ptr_id  = spv[pos+1];
      uint32_t mem_mask = spv[pos+3];
      if (mem_mask & 0x2u) { // Aligned bit
        auto it = id_psb_type.find(ptr_id);
        if (it != id_psb_type.end()) {
          auto bit = psb_ptr_base.find(it->second);
          if (bit != psb_ptr_base.end()) {
            uint32_t req = typeAlign(bit->second, info);
            if (req > spv[pos+4]) spv[pos+4] = req;
          }
        }
      }
    }
    // OpLoad: type=word[1], result=word[2], ptr=word[3], [mem_mask=word[4], align=word[5]]
    if (op == Op::Load && wc >= 6) {
      uint32_t ptr_id   = spv[pos+3];
      uint32_t mem_mask = spv[pos+4];
      if (mem_mask & 0x2u) { // Aligned bit
        auto it = id_psb_type.find(ptr_id);
        if (it != id_psb_type.end()) {
          auto bit = psb_ptr_base.find(it->second);
          if (bit != psb_ptr_base.end()) {
            uint32_t req = typeAlign(bit->second, info);
            if (req > spv[pos+5]) spv[pos+5] = req;
          }
        }
      }
    }
    pos += wc;
  }
}

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
      err = "module uses Intel-only SPIR-V capability " + std::to_string(cap);
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
        // base type, we'd hit VUID-Offset-04547 (Offset decorations forbidden
        // on Function-scope-used structs). Resolved via type cloning later in
        // emitVulkanSpirv: a clone struct id is allocated for PC use; the
        // original stays for the Function-scope local. No bail needed.
      }
    }
  }

  // If any function body uses OpExtInst with an import other than OpenCL.std
  // (e.g. GLSL.std.450 which we already produce), we can't translate those.
  // OpenCL.std calls are translated in-place during body emission via the
  // OpenCL.std → GLSL.std.450 / core SPIR-V mapping. Bail only if the
  // module uses an ext import we don't recognize OR uses an OpenCL.std
  // opcode we don't yet translate (the latter check happens in the body
  // emit path).
  if (!info.ext_inst_import_ids.empty()) {
    for (auto& [fn_id, fn] : info.functions) {
      for (auto& instr : fn.body) {
        if (instr.opcode == Op::ExtInst && instr.wc() >= 4) {
          uint32_t import = instr.word(3);
          if (info.ext_inst_import_ids.count(import) &&
              import != info.opencl_std_import_id) {
            auto nit = info.ext_inst_import_names.find(import);
            std::string n = (nit != info.ext_inst_import_names.end())
                              ? nit->second : "<unknown>";
            err = "module body uses OpExtInst from unsupported ext import: " + n;
            return {};
          }
        }
      }
    }
  }

  IdAllocator idAlloc(info.bound + 1);

  // -------------------------------------------------------------------------
  // Allocate IDs for new constructs
  // -------------------------------------------------------------------------

  // GLSL.std.450 ext-inst import (always allocated; emitted only if used).
  uint32_t glsl_std_450_id = idAlloc.alloc();
  bool glsl_std_450_used = false;

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

  // For each ByVal struct param whose struct type is also used as the base
  // type of a Function-scope pointer in the original module: allocate a
  // CLONE struct id. The clone goes into the PushConstant Block (where its
  // members get Offset decorations); the original stays for the Function-scope
  // local variable in the stub. Conversion clone→original happens in the
  // stub via OpCompositeExtract/Construct.
  for (auto& [stub_id, kpc] : kernel_pc_info) {
    for (auto& al : kpc.layout) {
      if (!al.is_byval) continue;
      uint32_t orig = al.type_id;
      bool used_in_function_scope = false;
      for (auto& [tid, pti] : info.ptr_types) {
        if (pti.base_type == orig && pti.storage_class == SC::Function) {
          used_in_function_scope = true;
          break;
        }
      }
      if (!used_in_function_scope) continue;
      // Allocate clone id and register it as a struct with the same members.
      // Subsequent code (PC type emission, decoration cascade, etc.) will
      // treat the clone like any other struct — but the PC layout entry now
      // points at it instead of the original.
      uint32_t clone_id = idAlloc.alloc();
      al.clone_type_id = clone_id;
      auto sit = info.struct_members.find(orig);
      if (sit != info.struct_members.end()) {
        info.struct_members[clone_id] = sit->second;
      }
      auto omit = info.member_offsets.find(orig);
      if (omit != info.member_offsets.end()) {
        info.member_offsets[clone_id] = omit->second;
      }
    }
  }

  // Helper: get the type id to use in PushConstant Block context for an
  // ArgLayout. For ByVal struct args that needed cloning (because the
  // original is also Function-scope-used), this returns the clone id;
  // otherwise the original type id.
  auto pcLayoutType = [](const ArgLayout& al) -> uint32_t {
    return al.clone_type_id != 0 ? al.clone_type_id : al.type_id;
  };

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
  // Pre-populate with existing function types so we reuse them instead of
  // emitting duplicates (duplicate non-aggregate types are forbidden in SPIR-V).
  std::map<std::vector<uint32_t>, uint32_t> new_fn_type_by_sig;
  for (auto& [fid, ret] : info.fn_types) {
    std::vector<uint32_t> sig = {ret};
    auto pit = info.fn_type_params.find(fid);
    if (pit != info.fn_type_params.end())
      for (uint32_t p : pit->second) sig.push_back(p);
    new_fn_type_by_sig.emplace(sig, fid); // only insert if not already present
  }

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

  // Generic typed-constant allocator. Returns an id for a constant of the
  // given (integer) type and value, reusing an existing constant from the
  // input module if one exists, otherwise allocating a fresh id and queuing
  // it for emission alongside refl constants.
  // Stored as map<(type_id, value), id> so we can emit each with its real type.
  std::map<std::pair<uint32_t,uint64_t>, uint32_t> extra_typed_const_ids;
  auto getTypedConst = [&](uint32_t type_id, uint64_t value) -> uint32_t {
    auto key = std::make_pair(type_id, value);
    auto it = extra_typed_const_ids.find(key);
    if (it != extra_typed_const_ids.end()) return it->second;
    for (auto& [cid, ci] : info.constants) {
      if (ci.type_id == type_id && ci.value == value) {
        extra_typed_const_ids[key] = cid;
        return cid;
      }
    }
    uint32_t id = idAlloc.alloc();
    extra_typed_const_ids[key] = id;
    return id;
  };

  // -------------------------------------------------------------------------
  // OpenCL.std → Vulkan-compatible OpExtInst translation pre-pass.
  // Walks every function body and rewrites OpExtInst calls into the
  // OpenCL.std import set as either core SPIR-V instructions or
  // GLSL.std.450 extended instructions (which are valid in Vulkan).
  // Bails (with a descriptive err message) for opcodes we don't yet
  // translate so the rest of the pipeline doesn't try to emit invalid
  // Vulkan SPIR-V. Must run BEFORE the header emit so the
  // OpExtInstImport "GLSL.std.450" is correctly gated.
  // -------------------------------------------------------------------------
  if (info.opencl_std_import_id != 0) {
    auto makeI = [](uint32_t opcode, std::initializer_list<uint32_t> ops) {
      Instr i;
      i.opcode = opcode;
      i.words = makeInstr(opcode, ops);
      return i;
    };
    for (auto& [fn_id, fn] : info.functions) {
      std::vector<Instr> new_body;
      new_body.reserve(fn.body.size());
      bool ok = true;
      for (auto& instr : fn.body) {
        if (instr.opcode == Op::ExtInst && instr.wc() >= 5 &&
            instr.word(3) == info.opencl_std_import_id) {
          uint32_t result_type = instr.word(1);
          uint32_t result_id   = instr.word(2);
          uint32_t ocl_op      = instr.word(4);
          // Operands start at word 5.
          auto op = [&](size_t i) { return instr.word(5 + i); };
          size_t nops = (instr.wc() > 5) ? (instr.wc() - 5) : 0;

          // Direct one-to-one GLSL.std.450 mappings: same operand layout,
          // just substitute the import set id and the ext-inst opcode.
          auto emit_glsl = [&](uint32_t glsl_op) {
            Words ops = {result_type, result_id, glsl_std_450_id, glsl_op};
            for (size_t i = 0; i < nops; ++i) ops.push_back(op(i));
            Instr ni;
            ni.opcode = Op::ExtInst;
            ni.words = makeInstr(Op::ExtInst, ops);
            new_body.push_back(std::move(ni));
            glsl_std_450_used = true;
          };

          switch (ocl_op) {
            // Math (float): direct GLSL.std.450 equivalents.
            case 14: emit_glsl(14); continue; // cos -> Cos
            case 15: emit_glsl(20); continue; // cosh -> Cosh
            case 19: emit_glsl(27); continue; // exp -> Exp
            case 20: emit_glsl(29); continue; // exp2 -> Exp2
            case 23: emit_glsl(4);  continue; // fabs -> FAbs
            case 25: emit_glsl(8);  continue; // floor -> Floor
            case 26: emit_glsl(50); continue; // fma -> Fma
            case 27: emit_glsl(40); continue; // fmax -> FMax
            case 28: emit_glsl(37); continue; // fmin -> FMin
            case 37: emit_glsl(28); continue; // log -> Log
            case 38: emit_glsl(30); continue; // log2 -> Log2
            case 42: emit_glsl(50); continue; // mad (a*b+c) ~= Fma
            case 48: emit_glsl(26); continue; // pow -> Pow
            case 56: emit_glsl(32); continue; // rsqrt -> InverseSqrt
            case 57: emit_glsl(13); continue; // sin -> Sin
            case 61: emit_glsl(31); continue; // sqrt -> Sqrt
            case 62: emit_glsl(15); continue; // tan -> Tan
            case 66: emit_glsl(3);  continue; // trunc -> Trunc

            // Integer: direct GLSL.std.450 equivalents.
            case 141: emit_glsl(5);  continue; // s_abs -> SAbs
            case 149: emit_glsl(45); continue; // s_clamp -> SClamp
            case 150: emit_glsl(44); continue; // u_clamp -> UClamp
            case 156: emit_glsl(42); continue; // s_max -> SMax
            case 157: emit_glsl(41); continue; // u_max -> UMax
            case 158: emit_glsl(39); continue; // s_min -> SMin
            case 159: emit_glsl(38); continue; // u_min -> UMin

            case 151: { // clz: convert to (width-1) - FindUMsb(x)
              auto wit = info.int_widths.find(result_type);
              if (wit == info.int_widths.end()) {
                err = "clz: result type is not an integer";
                ok = false; break;
              }
              uint32_t width = wit->second;
              uint32_t x = op(0);
              auto extInstr = [&](uint32_t set_id, uint32_t set_op,
                                  uint32_t res_ty, uint32_t res_id, uint32_t arg) {
                Instr ni;
                ni.opcode = Op::ExtInst;
                ni.words  = makeInstr(Op::ExtInst,
                    Words{res_ty, res_id, set_id, set_op, arg});
                return ni;
              };
              if (width <= 32) {
                // 32-bit (or narrower): clz(x) = 31 - FindUMsb(x).
                uint32_t tmp = idAlloc.alloc();
                uint32_t cw  = getTypedConst(result_type, width - 1);
                new_body.push_back(extInstr(glsl_std_450_id, 75u,
                    result_type, tmp, x));
                new_body.push_back(makeI(Op::ISub,
                    {result_type, result_id, cw, tmp}));
                glsl_std_450_used = true;
                continue;
              } else if (width == 64) {
                // 64-bit clz polyfill: GLSL FindUMsb only supports 32-bit, so
                // split into hi/lo 32-bit halves, find_msb each, then select.
                //   hi = (uint32)(x >> 32), lo = (uint32)x
                //   hi_clz = 31 - FindUMsb(hi)
                //   lo_clz = 31 - FindUMsb(lo)
                //   r32    = hi == 0 ? lo_clz + 32 : hi_clz
                //   result = (ulong)r32
                if (info.uint32_type_id == 0 || info.bool_type_id == 0) {
                  err = "clz 64-bit: missing uint32 or bool type in module";
                  ok = false; break;
                }
                uint32_t U32 = info.uint32_type_id;
                uint32_t U64 = result_type; // ulong
                uint32_t Bo  = info.bool_type_id;
                uint32_t c_ulong_32 = getTypedConst(U64, 32);
                uint32_t c_uint_31  = getTypedConst(U32, 31);
                uint32_t c_uint_32  = getTypedConst(U32, 32);
                uint32_t c_uint_0   = getTypedConst(U32, 0);
                uint32_t hi64    = idAlloc.alloc();
                uint32_t hi32    = idAlloc.alloc();
                uint32_t lo32    = idAlloc.alloc();
                uint32_t hi_msb  = idAlloc.alloc();
                uint32_t lo_msb  = idAlloc.alloc();
                uint32_t hi_clz  = idAlloc.alloc();
                uint32_t lo_clz  = idAlloc.alloc();
                uint32_t lo_p32  = idAlloc.alloc();
                uint32_t hi_zero = idAlloc.alloc();
                uint32_t r32     = idAlloc.alloc();
                new_body.push_back(makeI(Op::ShiftRightLogical,
                    {U64, hi64, x, c_ulong_32}));
                new_body.push_back(makeI(Op::UConvert, {U32, hi32, hi64}));
                new_body.push_back(makeI(Op::UConvert, {U32, lo32, x}));
                new_body.push_back(extInstr(glsl_std_450_id, 75u, U32, hi_msb, hi32));
                new_body.push_back(extInstr(glsl_std_450_id, 75u, U32, lo_msb, lo32));
                new_body.push_back(makeI(Op::ISub, {U32, hi_clz, c_uint_31, hi_msb}));
                new_body.push_back(makeI(Op::ISub, {U32, lo_clz, c_uint_31, lo_msb}));
                new_body.push_back(makeI(Op::IAdd, {U32, lo_p32, lo_clz, c_uint_32}));
                new_body.push_back(makeI(Op::IEqual, {Bo, hi_zero, hi32, c_uint_0}));
                new_body.push_back(makeI(Op::Select, {U32, r32, hi_zero, lo_p32, hi_clz}));
                new_body.push_back(makeI(Op::UConvert, {U64, result_id, r32}));
                glsl_std_450_used = true;
                continue;
              } else {
                err = "clz: unsupported width " + std::to_string(width);
                ok = false; break;
              }
            }
            case 166: { // popcount -> core SPIR-V OpBitCount
              if (nops != 1) { err = "popcount needs 1 operand"; ok = false; break; }
              new_body.push_back(makeI(Op::BitCount,
                  {result_type, result_id, op(0)}));
              continue;
            }
            case 169: // s_mul24 -> SMul24: (a & 0xFFFFFF) * (b & 0xFFFFFF)
            case 170: // u_mul24 -> UMul24
            {
              if (nops != 2) { err = "mul24 needs 2 operands"; ok = false; break; }
              auto wit = info.int_widths.find(result_type);
              if (wit == info.int_widths.end()) {
                err = "mul24: result type is not an integer";
                ok = false; break;
              }
              uint32_t mask = getTypedConst(result_type, 0xFFFFFFu);
              uint32_t a_lo = idAlloc.alloc();
              uint32_t b_lo = idAlloc.alloc();
              new_body.push_back(makeI(Op::BitwiseAnd, {result_type, a_lo, op(0), mask}));
              new_body.push_back(makeI(Op::BitwiseAnd, {result_type, b_lo, op(1), mask}));
              new_body.push_back(makeI(Op::IMul, {result_type, result_id, a_lo, b_lo}));
              continue;
            }
            case 167: // s_mad24 -> (a & 0xFFFFFF) * (b & 0xFFFFFF) + c
            case 168: // u_mad24
            {
              if (nops != 3) { err = "mad24 needs 3 operands"; ok = false; break; }
              auto wit = info.int_widths.find(result_type);
              if (wit == info.int_widths.end()) {
                err = "mad24: result type is not an integer";
                ok = false; break;
              }
              uint32_t mask = getTypedConst(result_type, 0xFFFFFFu);
              uint32_t a_lo = idAlloc.alloc();
              uint32_t b_lo = idAlloc.alloc();
              uint32_t prod = idAlloc.alloc();
              new_body.push_back(makeI(Op::BitwiseAnd, {result_type, a_lo, op(0), mask}));
              new_body.push_back(makeI(Op::BitwiseAnd, {result_type, b_lo, op(1), mask}));
              new_body.push_back(makeI(Op::IMul, {result_type, prod, a_lo, b_lo}));
              new_body.push_back(makeI(Op::IAdd, {result_type, result_id, prod, op(2)}));
              continue;
            }
            default:
              err = "untranslated OpenCL.std opcode " + std::to_string(ocl_op);
              ok = false;
              break;
          }
          if (!ok) break;
        }
        new_body.push_back(instr);
      }
      if (!ok) return {};
      fn.body = std::move(new_body);
    }
  }

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

  // --- OpExtInstImport GLSL.std.450 (only if used by translated OpenCL.std calls) ---
  if (glsl_std_450_used) {
    Words ops = {glsl_std_450_id};
    Words strWords = encodeLiteralString("GLSL.std.450");
    for (uint32_t w : strWords) ops.push_back(w);
    emitInstr(out, Op::ExtInstImport, ops);
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
    // Add all Workgroup-class global variables: shared memory vars are not in the
    // original interface list (OpenCL doesn't require it) but Vulkan SPIR-V 1.4+
    // requires all statically-used global vars to be listed.
    for (auto& orig_instr : info.original_types) {
      if (orig_instr.opcode == Op::Variable && orig_instr.wc() >= 4 &&
          orig_instr.word(3) == SC::WorkgroupLocal) {
        uint32_t var_id = orig_instr.word(2);
        ops.push_back(var_id);
      }
    }
    emitInstr(out, Op::EntryPoint, ops);
  }

  // Emit OpExecutionModeId LocalSizeId for each entry point.
  // This is required for GLCompute kernels with spec-constant workgroup size.
  // LocalSizeId (38) uses OpExecutionModeId (331) with spec constant IDs as operands.
  static const uint32_t kLocalSizeId = 38;
  for (auto& ep : info.entry_points) {
    emitInstr(out, Op::ExecutionModeId, {ep.fn_id, kLocalSizeId, wgs_x_id, wgs_y_id, wgs_z_id});
  }

  // WorkgroupSize is also specified via BuiltIn WorkgroupSize on the SpecConstantComposite (clspv style).

  // --- OpSource ---
  // Emit a simple source annotation
  emitInstr(out, Op::Source, {
    (uint32_t)spv::SourceLanguageOpenCL_C,
    120 // version 1.2
  });

  // Pre-scan: collect phantom result IDs — result IDs that are remapped to
  // another ID during function body transforms without being emitted as an
  // instruction result. These must be excluded from OpName to avoid
  // "forward referenced IDs" validation errors.
  std::unordered_set<uint32_t> phantom_result_ids;
  for (auto& [fn_id, fn] : info.functions) {
    for (auto& instr : fn.body) {
      // PtrCastToGeneric / GenericCastToPtr: always remapped, never emitted
      if ((instr.opcode == Op::PtrCastToGeneric || instr.opcode == Op::GenericCastToPtr)
          && instr.wc() >= 3) {
        phantom_result_ids.insert(instr.word(2));
        continue;
      }
      // InBoundsPtrAccessChain with zero elem index (non-CW/Generic type):
      // remapped to base ptr, no instruction emitted for result_id.
      if (instr.opcode == Op::InBoundsPtrAccessChain && instr.wc() >= 5) {
        uint32_t result_type = instr.word(1);
        uint32_t result_id   = instr.word(2);
        // CW/Generic types get PSB treatment → result IS emitted (as PtrAccessChain)
        if (cw_type_remap.count(result_type)) continue;
        uint32_t elem_idx_id = instr.word(4);
        bool has_extra_indices = (instr.wc() >= 6);
        auto cit = info.constants.find(elem_idx_id);
        if (cit != info.constants.end() && cit->second.value == 0 && !has_extra_indices)
          phantom_result_ids.insert(result_id);
        continue;
      }
      // Load from WorkgroupSize var: result is phantom when NOT v3ulong (which
      // would trigger an emitted UConvert); non-v3ulong case uses remap[]=.
      if (instr.opcode == Op::Load && instr.wc() >= 4 &&
          instr.word(3) == info.workgroup_size_var_id) {
        uint32_t result_type = instr.word(1);
        uint32_t result_id   = instr.word(2);
        auto vit = info.vec_types.find(result_type);
        bool is_v3ulong = (vit != info.vec_types.end() &&
                           vit->second.first == info.uint64_type_id &&
                           vit->second.second == 3);
        if (!is_v3ulong)
          phantom_result_ids.insert(result_id);
        continue;
      }
      // ConvertPtrToU for a CW pod_kind param in an impl fn: result remapped to
      // the param itself (stays as ulong), no instruction emitted.
      if (instr.opcode == Op::ConvertPtrToU && instr.wc() >= 4 &&
          info.impl_fn_ids.count(fn_id)) {
        uint32_t result_id = instr.word(2);
        uint32_t src_id    = instr.word(3);
        for (auto& p : fn.params) {
          if (p.is_crossworkgroup_ptr && p.id == src_id && !p.is_pointer_kind) {
            phantom_result_ids.insert(result_id);
            break;
          }
        }
      }
    }
  }

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
  // Phantom result IDs: remapped away during function body transform, never defined
  removed_ids.insert(phantom_result_ids.begin(), phantom_result_ids.end());

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
        // Decorate the PC-side type (clone for ByVal-conflict cases). This
        // keeps Offset decorations off the original Function-scope-used
        // struct id, satisfying VUID-Offset-04547.
        uint32_t pc_t = pcLayoutType(al);
        if (info.struct_members.count(pc_t))
          ensureStructOffsets(pc_t);
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

  // Note: Do NOT decorate wgs_composite_id with BuiltIn WorkgroupSize.
  // Per SPIR-V spec, BuiltIn WorkgroupSize supersedes LocalSizeId execution mode.
  // We use OpExecutionModeId LocalSizeId exclusively (emitted above with OpEntryPoint).
  // Keeping both would cause the BuiltIn decoration to override LocalSizeId.

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
  // Mark impl fn type IDs to skip (we'll emit new ones, or reuse existing ones).
  // Skip the original type whenever a new type is assigned (even if the new type
  // is an existing type from the module) — the original type may reference CW
  // pointer types that are being skipped, and is no longer needed.
  for (auto& [fn_id, new_type_id] : impl_fn_new_type_ids) {
    auto fit = info.functions.find(fn_id);
    if (fit == info.functions.end()) continue;
    uint32_t orig_type = fit->second.fn_type;
    if (new_type_id != orig_type)
      skip_types.insert(orig_type);
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
      // Valid memory-semantics values only use SPIR-V MemorySemanticsMask bits:
      //   0x002 Acquire, 0x004 Release, 0x008 AcquireRelease,
      //   0x010 SequentiallyConsistent,
      //   0x040 UniformMemory, 0x080 SubgroupMemory, 0x100 WorkgroupMemory,
      //   0x200 CrossWorkgroupMemory, 0x400 AtomicCounterMemory, 0x800 ImageMemory
      // Bits 0 (0x001) and 5 (0x020) are NOT valid MemorySemantics bits.
      // Additionally, the memory-order bits (Acquire/Release/AcqRel/SeqCst) are
      // mutually exclusive, so a valid value has EXACTLY ONE order bit set.
      // This rejects integer constants like 123 (0x7B, multiple order bits) and
      // 222 (0xDE, all order bits) that happen to have bit 0x10 set by coincidence.
      static const uint32_t kValidMemSemBits = 0xFDEu;
      uint32_t order_bits = value & 0x1Eu; // bits 1-4: order flags
      bool has_one_order_bit = (order_bits != 0) && ((order_bits & (order_bits - 1)) == 0);
      // Detect memory-semantics-looking constants and rewrite OpenCL-only
      // bits to Vulkan-compatible ones:
      //   SequentiallyConsistent (0x010) → AcquireRelease (0x008)
      //   CrossWorkgroupMemory   (0x200) → UniformMemory   (0x040)
      // Match: only uses valid mem-sem bits, has at least one memory-scope
      // bit, and has at most one order bit.
      if (is_uint32 && !(value & ~kValidMemSemBits) && (value & 0xFC0u) && has_one_order_bit) {
        uint32_t fixed = value;
        if (fixed & 0x010u) fixed = (fixed & ~0x010u) | 0x008u;
        if (fixed & 0x200u) fixed = (fixed & ~0x200u) | 0x040u;
        if (fixed != value) {
          emitInstr(out, Op::Constant, {type_id, const_id, fixed});
          continue;
        }
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
    // First emit any clone struct types that this kpc references, so the PC
    // struct's TypeStruct can name them. Members of the clone are identical
    // to the original (we just need a fresh struct id we can decorate).
    for (auto& al : kpc.layout) {
      if (al.clone_type_id != 0) {
        auto sit = info.struct_members.find(al.clone_type_id);
        if (sit != info.struct_members.end()) {
          Words clone_words = {al.clone_type_id};
          for (uint32_t mt : sit->second) clone_words.push_back(mt);
          emitInstr(out, Op::TypeStruct, clone_words);
        }
      }
    }

    // PC struct: one member per arg (uses clone for ByVal-conflict cases).
    Words pc_struct_words = {kpc.pc_struct_type_id};
    for (auto& al : kpc.layout) {
      pc_struct_words.push_back(pcLayoutType(al));
    }
    emitInstr(out, Op::TypeStruct, pc_struct_words);

    // ptr PushConstant to PC struct
    emitInstr(out, Op::TypePointer, {kpc.ptr_pc_type_id, SC::PushConstant, kpc.pc_struct_type_id});

    // ptr PushConstant to each member type (for OpAccessChain). Uses the
    // PC-side type (clone for ByVal-conflict cases) so AccessChain into PC
    // hands back a pointer with Block-decorated layout semantics.
    for (size_t i = 0; i < kpc.layout.size(); ++i) {
      emitInstr(out, Op::TypePointer, {kpc.member_ptr_type_ids[i], SC::PushConstant, pcLayoutType(kpc.layout[i])});
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
    if (info.fn_types.count(new_type_id)) continue; // reusing existing type — already in original_types
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
    if (info.fn_types.count(new_type_id)) continue; // reusing existing type
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

  // New typed constants from OpExtInst translation (clz needs e.g. ulong 63).
  for (auto& [key, id] : extra_typed_const_ids) {
    if (info.constants.count(id)) continue;
    uint32_t type_id = key.first;
    uint64_t value   = key.second;
    auto wit = info.int_widths.find(type_id);
    uint32_t width = (wit != info.int_widths.end()) ? wit->second : 32;
    if (width <= 32) {
      emitInstr(out, Op::Constant, {type_id, id, (uint32_t)value});
    } else {
      // 64-bit constant: two literal words.
      emitInstr(out, Op::Constant, {
        type_id, id,
        (uint32_t)(value & 0xFFFFFFFFu),
        (uint32_t)((value >> 32) & 0xFFFFFFFFu)
      });
    }
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

  // Map from global variable ID → storage class (for detecting Workgroup bases
  // in InBoundsPtrAccessChain transforms).
  std::unordered_map<uint32_t, uint32_t> global_var_sc;
  for (auto& orig_instr : info.original_types) {
    if (orig_instr.opcode == Op::Variable && orig_instr.wc() >= 4) {
      global_var_sc[orig_instr.word(2)] = orig_instr.word(3);
    }
  }
  // Also include chip_var (CrossWorkgroup) and other special vars
  for (auto& cv : info.chip_vars) {
    global_var_sc[cv.id] = SC::CrossWorkgroup;
  }

  // Helper: find Workgroup pointer type for a given base type, returns 0 if not found.
  auto findWorkgroupPtrType = [&](uint32_t base_type) -> uint32_t {
    for (auto& [tid, pi] : info.ptr_types) {
      if (pi.base_type == base_type && pi.storage_class == SC::WorkgroupLocal)
        return tid;
    }
    return 0;
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

        // Load. For ByVal-conflict cases, the PC member type is the CLONE
        // struct, so the load result type must also be the clone. For all
        // other cases (including non-byval), use the original type.
        uint32_t load_type = (al.is_byval && al.clone_type_id != 0)
                               ? al.clone_type_id : al.type_id;
        uint32_t load_id = idAlloc.alloc();
        uint32_t alignment = typeAlign(al.type_id, info);
        if (alignment == 0) alignment = 4;
        // OpLoad with alignment memory operand: Aligned = 0x2
        emitInstr(out, Op::Load, {load_type, load_id, chain_id, 0x2, alignment});

        if (al.is_byval) {
          // For byval-conflict cases, the loaded value has clone struct type
          // but the local Function-scope variable holds the original struct
          // type. Convert via member-by-member CompositeExtract +
          // CompositeConstruct (clone and original have identical members,
          // just different ids).
          uint32_t store_val = load_id;
          if (al.clone_type_id != 0) {
            auto sit = info.struct_members.find(al.type_id);
            if (sit != info.struct_members.end()) {
              std::vector<uint32_t> members;
              for (size_t j = 0; j < sit->second.size(); ++j) {
                uint32_t mt = sit->second[j];
                uint32_t mid = idAlloc.alloc();
                emitInstr(out, Op::CompositeExtract, {mt, mid, load_id, (uint32_t)j});
                members.push_back(mid);
              }
              uint32_t orig_val = idAlloc.alloc();
              Words ops = {al.type_id, orig_val};
              for (uint32_t m : members) ops.push_back(m);
              emitInstr(out, Op::CompositeConstruct, ops);
              store_val = orig_val;
            }
          }
          // Store to local var
          uint32_t local_var = local_struct_var_ids[i];
          emitInstr(out, Op::Store, {local_var, store_val, 0x2, alignment});
          call_args.push_back(local_var);
        } else {
          // If impl param is bool but we loaded uint32, convert: bool = (uint32 != 0)
          uint32_t arg_id = load_id;
          if (i < impl_fn.params.size() &&
              impl_fn.params[i].type_id == info.bool_type_id &&
              al.type_id == info.uint32_type_id) {
            uint32_t bool_id = idAlloc.alloc();
            emitInstr(out, Op::INotEqual, {info.bool_type_id, bool_id, load_id, const_uint32_0_id});
            arg_id = bool_id;
          }
          call_args.push_back(arg_id);
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

          // Cannot represent CopyMemorySized in Vulkan SPIR-V (requires Addresses cap).
          err = "in-body unsupported pattern (clspv-fallback comment)"; return {};
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
            // If the base (after remap) is a Workgroup variable, we cannot produce a
            // PSB pointer — use Workgroup AccessChain semantics instead.
            auto base_sc_it = global_var_sc.find(base);
            if (base_sc_it != global_var_sc.end() &&
                base_sc_it->second == SC::WorkgroupLocal) {
              // Base is a Workgroup variable: use OpAccessChain with Workgroup ptr type.
              uint32_t wg_ptr_type = findWorkgroupPtrType(base_type);
              if (wg_ptr_type != 0) {
                bool ez = (instr.wc() >= 5 &&
                           [&]{ auto cit = info.constants.find(instr.word(4));
                                return cit != info.constants.end() && cit->second.value == 0; }());
                if (ez && instr.wc() >= 6) {
                  // elem_idx=0 + extra indices → AccessChain with extra indices
                  Words ops = {wg_ptr_type, result_id, base};
                  for (uint32_t i = 5; i < instr.wc(); ++i)
                    ops.push_back(remapId(instr.word(i), remap));
                  emitInstr(out, Op::AccessChain, ops);
                } else {
                  Words ops = {wg_ptr_type, result_id, base};
                  for (uint32_t i = 4; i < instr.wc(); ++i)
                    ops.push_back(remapId(instr.word(i), remap));
                  emitInstr(out, Op::PtrAccessChain, ops);
                }
                continue;
              }
            }
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
          bool has_extra_indices = (instr.wc() >= 6);
          if (elem_is_zero && !has_extra_indices) {
            // zero offset, no sub-indices: result == base pointer
            remap[result_id] = base;
          } else {
            // Only Workgroup SC is valid for PtrAccessChain in Vulkan.
            // Function/Private SC pointer arithmetic cannot be expressed — bail out.
            if (pit == info.ptr_types.end() ||
                pit->second.storage_class != SC::WorkgroupLocal) {
              err = "in-body unsupported pattern (subprocess fallback)"; return {};
            }
            if (elem_is_zero && has_extra_indices) {
              // elem_idx=0 is a no-op step; additional indices navigate into the
              // composite object. Use OpAccessChain (composite navigation) rather
              // than OpPtrAccessChain (pointer arithmetic with stride of base type).
              Words ops = {result_type, result_id, base};
              for (uint32_t i = 5; i < instr.wc(); ++i)
                ops.push_back(remapId(instr.word(i), remap));
              emitInstr(out, Op::AccessChain, ops);
            } else {
              uint32_t elem_idx = remapId(elem_idx_id, remap);
              Words ops = {result_type, result_id, base, elem_idx};
              for (uint32_t i = 5; i < instr.wc(); ++i)
                ops.push_back(remapId(instr.word(i), remap));
              emitInstr(out, Op::PtrAccessChain, ops);
            }
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
              // wgs_composite_id is v3uint SpecConstantComposite.
              // Avoid OpUConvert of a SpecConstantComposite (driver portability issue):
              // extract each component and scalar-convert to ulong, then reconstruct.
              uint32_t cx = idAlloc.alloc(), cy = idAlloc.alloc(), cz = idAlloc.alloc();
              uint32_t lx = idAlloc.alloc(), ly = idAlloc.alloc(), lz = idAlloc.alloc();
              emitInstr(out, Op::CompositeExtract, {info.uint32_type_id, cx, wgs_composite_id, 0});
              emitInstr(out, Op::CompositeExtract, {info.uint32_type_id, cy, wgs_composite_id, 1});
              emitInstr(out, Op::CompositeExtract, {info.uint32_type_id, cz, wgs_composite_id, 2});
              emitInstr(out, Op::UConvert, {info.uint64_type_id, lx, cx});
              emitInstr(out, Op::UConvert, {info.uint64_type_id, ly, cy});
              emitInstr(out, Op::UConvert, {info.uint64_type_id, lz, cz});
              emitInstr(out, Op::CompositeConstruct, {result_type, result_id, lx, ly, lz});
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
              // UConvert tmp (v3uint) → result_id (v3ulong) component-wise to avoid
              // driver issues with vector UConvert.
              uint32_t cx = idAlloc.alloc(), cy = idAlloc.alloc(), cz = idAlloc.alloc();
              uint32_t lx = idAlloc.alloc(), ly = idAlloc.alloc(), lz = idAlloc.alloc();
              emitInstr(out, Op::CompositeExtract, {info.uint32_type_id, cx, tmp_id, 0});
              emitInstr(out, Op::CompositeExtract, {info.uint32_type_id, cy, tmp_id, 1});
              emitInstr(out, Op::CompositeExtract, {info.uint32_type_id, cz, tmp_id, 2});
              emitInstr(out, Op::UConvert, {info.uint64_type_id, lx, cx});
              emitInstr(out, Op::UConvert, {info.uint64_type_id, ly, cy});
              emitInstr(out, Op::UConvert, {info.uint64_type_id, lz, cz});
              emitInstr(out, Op::CompositeConstruct, {result_type, result_id, lx, ly, lz});
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
              // Pre-scan missed this type — indicates a bug in collectModuleInfo.
              err = "in-body unsupported pattern (clspv-fallback comment)"; return {};
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

        // OpAtomic* with pointer operand at word[3] and chip_var as pointer →
        // AccessChain + atomic on the chained pointer. Covers the common
        // result-producing atomics (IAdd, ISub, IIncrement, IDecrement, And,
        // Or, Xor, SMin/UMin/SMax/UMax, Exchange) and OpAtomicLoad.
        // OpAtomicStore (228) has the pointer at word[1] (no result), handled
        // separately below.
        // The pointer may be the chip_var directly OR an OpPtrCastToGeneric
        // result that earlier got remapped to the chip_var id.
        if (instr.opcode >= 227 && instr.opcode <= 242 &&
            instr.opcode != 228 /*Store*/ && instr.wc() >= 4) {
          uint32_t ptr_id = remapId(instr.word(3), remap);
          auto cv_it = info.chip_var_by_id.find(ptr_id);
          if (cv_it != info.chip_var_by_id.end()) {
            ChipVarInfo& cv = info.chip_vars[cv_it->second];
            uint32_t sb_ptr_type = ptr_sb_type_map.count(cv.base_type_id)
                                   ? ptr_sb_type_map[cv.base_type_id] : 0;
            if (sb_ptr_type != 0) {
              uint32_t chain_id = idAlloc.alloc();
              emitInstr(out, Op::AccessChain, {sb_ptr_type, chain_id, cv.ssbo_var_id, const_uint32_0_id});
              Words atomic_words = {instr.word(1), instr.word(2), chain_id};
              for (uint32_t i = 4; i < instr.wc(); ++i)
                atomic_words.push_back(remapId(instr.word(i), remap));
              emitInstr(out, instr.opcode, atomic_words);
              continue;
            }
          }
        }
        if (instr.opcode == 228 /*OpAtomicStore*/ && instr.wc() >= 4) {
          uint32_t ptr_id = remapId(instr.word(1), remap);
          auto cv_it = info.chip_var_by_id.find(ptr_id);
          if (cv_it != info.chip_var_by_id.end()) {
            ChipVarInfo& cv = info.chip_vars[cv_it->second];
            uint32_t sb_ptr_type = ptr_sb_type_map.count(cv.base_type_id)
                                   ? ptr_sb_type_map[cv.base_type_id] : 0;
            if (sb_ptr_type != 0) {
              uint32_t chain_id = idAlloc.alloc();
              emitInstr(out, Op::AccessChain, {sb_ptr_type, chain_id, cv.ssbo_var_id, const_uint32_0_id});
              Words store_words = {chain_id};
              for (uint32_t i = 2; i < instr.wc(); ++i)
                store_words.push_back(remapId(instr.word(i), remap));
              emitInstr(out, instr.opcode, store_words);
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
          bool has_extra_indices = (instr.wc() >= 6);
          if (elem_is_zero && !has_extra_indices) {
            remap[result_id] = base;
          } else {
            // PtrAccessChain is only valid for PSB, Workgroup, StorageBuffer SC.
            // If not remapped to PSB and not Workgroup/StorageBuffer, bail out.
            if (!was_remapped) {
              auto orig_pit = info.ptr_types.find(orig_type);
              if (orig_pit == info.ptr_types.end() ||
                  (orig_pit->second.storage_class != SC::WorkgroupLocal &&
                   orig_pit->second.storage_class != SC::StorageBuffer)) {
                err = "in-body unsupported pattern (subprocess fallback)"; return {};
              }
            }
            if (elem_is_zero && has_extra_indices && !was_remapped) {
              // elem_idx=0 is a no-op step; extra indices navigate into the composite.
              Words ops = {result_type, result_id, base};
              for (uint32_t i = 5; i < instr.wc(); ++i)
                ops.push_back(remapId(instr.word(i), remap));
              emitInstr(out, Op::AccessChain, ops);
            } else {
              uint32_t elem_idx = remapId(elem_idx_id, remap);
              Words ops = {result_type, result_id, base, elem_idx};
              for (uint32_t i = 5; i < instr.wc(); ++i)
                ops.push_back(remapId(instr.word(i), remap));
              emitInstr(out, Op::PtrAccessChain, ops);
            }
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

          // Cannot represent CopyMemorySized in Vulkan SPIR-V (requires Addresses cap).
          err = "in-body unsupported pattern (clspv-fallback comment)"; return {};
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
              uint32_t cx = idAlloc.alloc(), cy = idAlloc.alloc(), cz = idAlloc.alloc();
              uint32_t lx = idAlloc.alloc(), ly = idAlloc.alloc(), lz = idAlloc.alloc();
              emitInstr(out, Op::CompositeExtract, {info.uint32_type_id, cx, wgs_composite_id, 0});
              emitInstr(out, Op::CompositeExtract, {info.uint32_type_id, cy, wgs_composite_id, 1});
              emitInstr(out, Op::CompositeExtract, {info.uint32_type_id, cz, wgs_composite_id, 2});
              emitInstr(out, Op::UConvert, {info.uint64_type_id, lx, cx});
              emitInstr(out, Op::UConvert, {info.uint64_type_id, ly, cy});
              emitInstr(out, Op::UConvert, {info.uint64_type_id, lz, cz});
              emitInstr(out, Op::CompositeConstruct, {result_type, result_id, lx, ly, lz});
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
              uint32_t cx = idAlloc.alloc(), cy = idAlloc.alloc(), cz = idAlloc.alloc();
              uint32_t lx = idAlloc.alloc(), ly = idAlloc.alloc(), lz = idAlloc.alloc();
              emitInstr(out, Op::CompositeExtract, {info.uint32_type_id, cx, tmp_id, 0});
              emitInstr(out, Op::CompositeExtract, {info.uint32_type_id, cy, tmp_id, 1});
              emitInstr(out, Op::CompositeExtract, {info.uint32_type_id, cz, tmp_id, 2});
              emitInstr(out, Op::UConvert, {info.uint64_type_id, lx, cx});
              emitInstr(out, Op::UConvert, {info.uint64_type_id, ly, cy});
              emitInstr(out, Op::UConvert, {info.uint64_type_id, lz, cz});
              emitInstr(out, Op::CompositeConstruct, {result_type, result_id, lx, ly, lz});
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

        // InBoundsPtrAccessChain is invalid in Vulkan (requires Addresses cap).
        // Convert to PtrAccessChain whenever the result type maps (or already is)
        // a Vulkan-valid SC (PSB, Workgroup, StorageBuffer). This branch covers
        // entry-point kernels that aren't a stub-impl pair (e.g. inline kernels).
        if (instr.opcode == Op::InBoundsPtrAccessChain && instr.wc() >= 5) {
          uint32_t orig_type   = instr.word(1);
          uint32_t result_type = orig_type;
          uint32_t result_id   = instr.word(2);
          uint32_t base        = remapId(instr.word(3), remap);
          auto cwit = cw_type_remap.find(result_type);
          bool was_remapped = (cwit != cw_type_remap.end());
          if (was_remapped) result_type = cwit->second;
          uint32_t elem_idx_id = instr.word(4);
          auto cit = info.constants.find(elem_idx_id);
          bool elem_is_zero = (cit != info.constants.end() && cit->second.value == 0);
          bool has_extra_indices = (instr.wc() >= 6);
          if (elem_is_zero && !has_extra_indices) {
            remap[result_id] = base;
          } else {
            if (!was_remapped) {
              auto orig_pit = info.ptr_types.find(orig_type);
              if (orig_pit == info.ptr_types.end() ||
                  (orig_pit->second.storage_class != SC::WorkgroupLocal &&
                   orig_pit->second.storage_class != SC::StorageBuffer &&
                   orig_pit->second.storage_class != SC::PhysicalStorageBuffer)) {
                err = "in-body unsupported pattern (clvk pipeline)"; return {};
              }
            }
            if (elem_is_zero && has_extra_indices && !was_remapped) {
              Words ops = {result_type, result_id, base};
              for (uint32_t i = 5; i < instr.wc(); ++i)
                ops.push_back(remapId(instr.word(i), remap));
              emitInstr(out, Op::AccessChain, ops);
            } else {
              uint32_t elem_idx = remapId(elem_idx_id, remap);
              Words ops = {result_type, result_id, base, elem_idx};
              for (uint32_t i = 5; i < instr.wc(); ++i)
                ops.push_back(remapId(instr.word(i), remap));
              emitInstr(out, Op::PtrAccessChain, ops);
            }
          }
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

  // Fix OpStore/OpLoad alignments for PSB pointers (VUID-06314).
  fixPSBAlignments(out, info);

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

  return result;
}

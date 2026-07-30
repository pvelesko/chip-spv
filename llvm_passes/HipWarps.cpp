//===- HipWarps.cpp -.-----------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM IR pass to handle kernels that are sensitive to warp width.
//
// (c) 2022-2023 Pekka Jääskeläinen / Intel
//===----------------------------------------------------------------------===//
//
// Currently handles kernels that call warp primitives that rely on the
// known warp width by using the reqd_subgroup_size() kernel attribute.
//
// TODO:
// * Lock-step semantics: CUDA/HIP allows dropping explicit thread/WI
// synchronization for cases where warp lock-step semantics guarantees
// a well-defined read-modify-write interleaving inside the warp. We should
// add an annotation that guarantees subgroup lockstep semantics in that case.
// There is not such an OpenCL extension yet to my knowledge.
//===----------------------------------------------------------------------===//

#include "HipWarps.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Metadata.h>
#include "llvm/IR/Module.h"

#include "chipStarConfig.hh"

PreservedAnalyses HipWarpsPass::run(Module &Mod, ModuleAnalysisManager &AM) {

  // We emulate warps with subgroups of which size is implementation and
  // kernel-specific by default while in CUDA/HIP it's a device-specific
  // constant that can be queried from the device info.
  //
  // Add the intel_reqd_sub_group_size kernel metadata to force the subgroup
  // size to be fixed to the warp size used by the chipStar build.
  //
  // Every kernel is stamped, unconditionally. Device code is compiled against
  // a warpSize that is a compile-time constant (CHIP_DEFAULT_WARP_SIZE, see
  // include/hip/devicelib/sync_and_util.hh), so a kernel is warp-width
  // sensitive whenever it relies on that constant -- and that is not something
  // this pass can detect:
  //
  //  * Warp lock-step semantics. CUDA/HIP lets device code drop explicit
  //    synchronization whenever warp lock-step guarantees a well-defined
  //    read-modify-write interleaving inside the warp. Such code calls no
  //    warp primitive at all; it only indexes shared memory by
  //    threadIdx % warpSize. Kokkos' HIP block reduction
  //    (HIPReductionsFunctor<F,false>::scalar_intra_warp_reduction) is exactly
  //    this: a barrier-free binary tree over warpSize shared-memory slots
  //    followed by a barrier-free broadcast of the warp leader's slot. If the
  //    subgroup is narrower than warpSize the halves race and each
  //    contribution is accumulated several times over. See
  //    tests/runtime/TestWarpSyncSharedMemReduction.hip.
  //
  //  * Matching a declaration list cannot work either. The previous version of
  //    this pass returned early unless one of a fixed list of mangled warp
  //    primitive names appeared in the module. That check is order dependent:
  //    it holds per translation unit, but with -fgpu-rdc chipStar runs its
  //    passes on the device-linked module, by which point the primitives are
  //    resolved and inlined and no matching declaration is left. Every kernel
  //    in every RDC build therefore went unstamped, IGC was free to pick the
  //    subgroup size, and -cl-opt-disable (or merely enough register pressure)
  //    made it pick 16 while device code still believed warpSize was 32.
  //
  // Cost: kernels that would rather have been compiled narrower are now pinned
  // to the warp size, which IGC honours even at the price of spilling. That is
  // the correct trade -- a silently wrong reduction is worse than a slow one.

  // Kernels that perform an indirect call must not be stamped: the driver then
  // delivers the correct 'this' but zero for every argument after it, silently.
  // No error, no diagnostic, just wrong results.
  //
  // See tests/runtime/TestIndirectCall.hip.
  //
  // A kernel that both dispatches indirectly and depends on the warp width
  // cannot be served either way; correct arguments are the more useful half.
  auto reachesIndirectCall = [](Function &Kernel) {
    SmallPtrSet<Function *, 16> Seen;
    SmallVector<Function *, 16> Worklist{&Kernel};
    Seen.insert(&Kernel);
    while (!Worklist.empty()) {
      Function *F = Worklist.pop_back_val();
      for (Instruction &I : instructions(*F)) {
        auto *CB = dyn_cast<CallBase>(&I);
        if (!CB || CB->isInlineAsm())
          continue;
        Function *Callee = CB->getCalledFunction();
        if (!Callee)
          return true; // Calls through a value: a vtable slot, a callback, ...
        if (!Callee->isDeclaration() && Seen.insert(Callee).second)
          Worklist.push_back(Callee);
      }
    }
    return false;
  };

  auto &Ctx = Mod.getContext();
  for (auto &F : Mod) {
    if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;
    if (reachesIndirectCall(F))
      continue;

    IntegerType *I32Type = IntegerType::get(Ctx, 32);
    F.setMetadata("intel_reqd_sub_group_size",
                  MDNode::get(Ctx, ConstantAsMetadata::get(ConstantInt::get(
                                       I32Type, CHIP_DEFAULT_WARP_SIZE))));
  }

  // The metadata should not impact other chipStar passes.
  return PreservedAnalyses::all();
}

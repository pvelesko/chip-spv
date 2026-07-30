; ModuleID = 'TestWarpSyncSharedMemReduction-hip-spirv64-generic-link.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

%struct.Reducer = type { ptr addrspace(4) }

@llvm.compiler.used = appending addrspace(1) global [2 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) @__chipspv_abort_called to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @__hip_cuid_6f177de16cedf7f3 to ptr addrspace(4))], section "llvm.metadata"
@__chipspv_abort_called = weak hidden addrspace(1) externally_initialized global i32 0, align 4
@.str = private unnamed_addr addrspace(1) constant [47 x i8] c"%s:%u: %s: Device-side assertion `%s' failed.\0A\00", align 1
@.str.1 = private unnamed_addr addrspace(1) constant [43 x i8] c"Device-side pure virtual function called.\0A\00", align 1
@shared = external hidden addrspace(3) global [0 x double], align 8
@__hip_cuid_6f177de16cedf7f3 = addrspace(1) global i8 0
@__chipspv_device_heap = weak hidden local_unnamed_addr addrspace(1) externally_initialized global ptr addrspace(4) null, align 8

; Function Attrs: convergent mustprogress noinline nounwind
define weak hidden spir_func void @__assert_fail(ptr addrspace(4) noundef %assertion, ptr addrspace(4) noundef %file, i32 noundef %line, ptr addrspace(4) noundef %function) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func i32 (ptr addrspace(4), ...) @printf(ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @.str to ptr addrspace(4)), ptr addrspace(4) noundef %file, i32 noundef %line, ptr addrspace(4) noundef %function, ptr addrspace(4) noundef %assertion) #7
  tail call spir_func void @__chipspv_abort(ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @__chipspv_abort_called to ptr addrspace(4))) #7
  ret void
}

; Function Attrs: convergent nofree nounwind
declare hidden spir_func noundef i32 @printf(ptr addrspace(4) noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare hidden spir_func void @__chipspv_abort(ptr addrspace(4) noundef) local_unnamed_addr #2

; Function Attrs: convergent mustprogress noinline nounwind
define weak hidden spir_func void @__cxa_pure_virtual() local_unnamed_addr #0 {
entry:
  %call = tail call spir_func i32 (ptr addrspace(4), ...) @printf(ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @.str.1 to ptr addrspace(4))) #7
  tail call spir_func void @__chipspv_abort(ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @__chipspv_abort_called to ptr addrspace(4))) #7
  ret void
}

; Function Attrs: convergent mustprogress noinline nounwind
define hidden spir_func void @_Z18intraWarpReductionRK7ReducerPdbi(ptr addrspace(4) readnone align 8 captures(none) %functor, ptr addrspace(4) noundef captures(none) %value, i1 noundef zeroext %skipVector, i32 noundef %width) local_unnamed_addr #0 {
entry:
  %call.i14 = tail call spir_func i64 @_Z12get_local_idj(i32 noundef 1) #7
  %conv.i = trunc i64 %call.i14 to i32
  %call.i = tail call spir_func i64 @_Z14get_local_sizej(i32 noundef 0) #7
  %conv.i15 = trunc i64 %call.i to i32
  %mul = mul i32 %conv.i15, %conv.i
  %call.i18 = tail call spir_func i64 @_Z12get_local_idj(i32 noundef 0) #7
  %conv.i19 = trunc i64 %call.i18 to i32
  %add = add i32 %mul, %conv.i19
  %rem = and i32 %add, 31
  br i1 %skipVector, label %cond.true, label %cond.end

cond.true:                                        ; preds = %entry
  %call.i16 = tail call spir_func i64 @_Z14get_local_sizej(i32 noundef 0) #7
  %conv.i17 = trunc i64 %call.i16 to i32
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i32 [ %conv.i17, %cond.true ], [ 1, %entry ]
  %cmp20 = icmp slt i32 %cond, %width
  br i1 %cmp20, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.inc, %cond.end
  %idx.ext10 = zext nneg i32 %rem to i64
  %idx.neg = sub nsw i64 0, %idx.ext10
  %add.ptr11 = getelementptr inbounds double, ptr addrspace(4) %value, i64 %idx.neg
  %0 = load double, ptr addrspace(4) %add.ptr11, align 8, !tbaa !13
  store double %0, ptr addrspace(4) %value, align 8, !tbaa !13
  ret void

for.body:                                         ; preds = %cond.end, %for.inc
  %delta.021 = phi i32 [ %mul6, %for.inc ], [ %cond, %cond.end ]
  %add4 = add nsw i32 %delta.021, %rem
  %cmp5 = icmp slt i32 %add4, %width
  %mul6 = shl nsw i32 %delta.021, 1
  br i1 %cmp5, label %land.lhs.true, label %for.inc

land.lhs.true:                                    ; preds = %for.body
  %rem7 = srem i32 %rem, %mul6
  %cmp8 = icmp eq i32 %rem7, 0
  br i1 %cmp8, label %if.then, label %for.inc

if.then:                                          ; preds = %land.lhs.true
  %idx.ext = sext i32 %delta.021 to i64
  %add.ptr = getelementptr inbounds double, ptr addrspace(4) %value, i64 %idx.ext
  %1 = load double, ptr addrspace(4) %add.ptr, align 8, !tbaa !13
  %2 = load double, ptr addrspace(4) %value, align 8, !tbaa !13
  %add.i = fadd contract double %1, %2
  store double %add.i, ptr addrspace(4) %value, align 8, !tbaa !13
  br label %for.inc

for.inc:                                          ; preds = %for.body, %land.lhs.true, %if.then
  %cmp = icmp slt i32 %mul6, %width
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !15
}

; Function Attrs: convergent nounwind
declare hidden spir_func i64 @_Z12get_local_idj(i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent nounwind
declare hidden spir_func i64 @_Z14get_local_sizej(i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent noinline nounwind
define hidden spir_func void @_Z19intraBlockReductionRK7ReducerdPdS2_(ptr addrspace(4) noundef readonly align 8 captures(none) dereferenceable(8) %functor, double noundef %value, ptr addrspace(4) noundef writeonly captures(none) %result, ptr addrspace(4) noundef captures(none) %shared) local_unnamed_addr #3 {
entry:
  %call.i32 = tail call spir_func i64 @_Z12get_local_idj(i32 noundef 1) #7
  %conv.i = trunc i64 %call.i32 to i32
  %call.i = tail call spir_func i64 @_Z14get_local_sizej(i32 noundef 0) #7
  %conv.i47 = trunc i64 %call.i to i32
  %mul = mul i32 %conv.i47, %conv.i
  %call.i48 = tail call spir_func i64 @_Z14get_local_sizej(i32 noundef 0) #7
  %conv.i49 = trunc i64 %call.i48 to i32
  %call.i60 = tail call spir_func i64 @_Z14get_local_sizej(i32 noundef 1) #7
  %conv.i61 = trunc i64 %call.i60 to i32
  %mul4 = mul i32 %conv.i61, %conv.i49
  %call.i33 = tail call spir_func i64 @_Z12get_local_idj(i32 noundef 1) #7
  %call.i50 = tail call spir_func i64 @_Z14get_local_sizej(i32 noundef 0) #7
  %mul7 = mul i64 %call.i50, %call.i33
  %idx.ext = and i64 %mul7, 4294967295
  %add.ptr = getelementptr inbounds nuw double, ptr addrspace(4) %shared, i64 %idx.ext
  %call.i54 = tail call spir_func i64 @_Z12get_local_idj(i32 noundef 0) #7
  %idx.ext9 = and i64 %call.i54, 4294967295
  %add.ptr10 = getelementptr inbounds nuw double, ptr addrspace(4) %add.ptr, i64 %idx.ext9
  store double %value, ptr addrspace(4) %add.ptr10, align 8, !tbaa !13
  %cond = tail call i32 @llvm.umin.i32(i32 %mul4, i32 32)
  tail call spir_func void @_Z18intraWarpReductionRK7ReducerPdbi(ptr addrspace(4) align 8 poison, ptr addrspace(4) noundef %add.ptr10, i1 noundef zeroext true, i32 noundef %cond) #7
  %call.i35 = tail call spir_func i64 @_Z12get_local_idj(i32 noundef 1) #7
  %idxprom = and i64 %call.i35, 4294967295
  %arrayidx = getelementptr inbounds nuw double, ptr addrspace(4) %shared, i64 %idxprom
  %0 = load double, ptr addrspace(4) %arrayidx, align 8, !tbaa !13
  %1 = load ptr addrspace(4), ptr addrspace(4) %functor, align 8, !tbaa !17
  %call.i37 = tail call spir_func i64 @_Z12get_local_idj(i32 noundef 1) #7
  %idxprom13 = and i64 %call.i37, 4294967295
  %arrayidx14 = getelementptr inbounds nuw double, ptr addrspace(4) %1, i64 %idxprom13
  store double %0, ptr addrspace(4) %arrayidx14, align 8, !tbaa !13
  tail call spir_func void @_Z7barrierj(i32 noundef 1) #7
  %cmp15 = icmp ult i32 %mul, 32
  br i1 %cmp15, label %if.then, label %if.end41

if.then:                                          ; preds = %entry
  %call.i39 = tail call spir_func i64 @_Z12get_local_idj(i32 noundef 1) #7
  %conv.i40 = trunc i64 %call.i39 to i32
  %call.i52 = tail call spir_func i64 @_Z14get_local_sizej(i32 noundef 0) #7
  %conv.i53 = trunc i64 %call.i52 to i32
  %mul18 = mul i32 %conv.i53, %conv.i40
  %call.i56 = tail call spir_func i64 @_Z12get_local_idj(i32 noundef 0) #7
  %conv.i57 = trunc i64 %call.i56 to i32
  %add20 = add i32 %mul18, %conv.i57
  %mul21 = shl i32 %add20, 5
  %cmp22 = icmp ult i32 %mul21, %mul4
  br i1 %cmp22, label %if.then23, label %if.end

if.then23:                                        ; preds = %if.then
  %idxprom24 = zext i32 %mul21 to i64
  %arrayidx25 = getelementptr inbounds nuw double, ptr addrspace(4) %shared, i64 %idxprom24
  %2 = load double, ptr addrspace(4) %arrayidx25, align 8, !tbaa !13
  store double %2, ptr addrspace(4) %add.ptr10, align 8, !tbaa !13
  br label %if.end

if.end:                                           ; preds = %if.then23, %if.then
  %div2616 = lshr i32 %mul4, 5
  tail call spir_func void @_Z18intraWarpReductionRK7ReducerPdbi(ptr addrspace(4) align 8 poison, ptr addrspace(4) noundef %add.ptr10, i1 noundef zeroext false, i32 noundef %div2616) #7
  %call.i41 = tail call spir_func i64 @_Z12get_local_idj(i32 noundef 1) #7
  %idxprom28 = and i64 %call.i41, 4294967295
  %arrayidx29 = getelementptr inbounds nuw double, ptr addrspace(4) %shared, i64 %idxprom28
  %3 = load double, ptr addrspace(4) %arrayidx29, align 8, !tbaa !13
  %4 = load ptr addrspace(4), ptr addrspace(4) %functor, align 8, !tbaa !17
  %call.i43 = tail call spir_func i64 @_Z12get_local_idj(i32 noundef 1) #7
  %add32 = add i64 %call.i43, 128
  %idxprom33 = and i64 %add32, 4294967295
  %arrayidx34 = getelementptr inbounds nuw double, ptr addrspace(4) %4, i64 %idxprom33
  store double %3, ptr addrspace(4) %arrayidx34, align 8, !tbaa !13
  %call.i58 = tail call spir_func i64 @_Z12get_local_idj(i32 noundef 0) #7
  %conv.i59 = trunc i64 %call.i58 to i32
  %call.i45 = tail call spir_func i64 @_Z12get_local_idj(i32 noundef 1) #7
  %conv.i46 = trunc i64 %call.i45 to i32
  %add37 = sub i32 0, %conv.i46
  %cmp38 = icmp eq i32 %conv.i59, %add37
  br i1 %cmp38, label %if.then39, label %if.end41

if.then39:                                        ; preds = %if.end
  %5 = load double, ptr addrspace(4) %shared, align 8, !tbaa !13
  store double %5, ptr addrspace(4) %result, align 8, !tbaa !13
  br label %if.end41

if.end41:                                         ; preds = %if.end, %if.then39, %entry
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #4

; Function Attrs: convergent nounwind
declare spir_func void @_Z7barrierj(i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent norecurse nounwind
define hidden spir_kernel void @_Z14warpSyncReducePdS_(ptr addrspace(1) noundef %out.coerce, ptr addrspace(1) noundef %snap.coerce) local_unnamed_addr #5 {
entry:
  %functor = alloca %struct.Reducer, align 8
  %functor.ascast = addrspacecast ptr %functor to ptr addrspace(4)
  %0 = ptrtoint ptr addrspace(1) %snap.coerce to i64
  %1 = inttoptr i64 %0 to ptr addrspace(4)
  call void @llvm.lifetime.start.p0(ptr nonnull %functor) #8
  store ptr addrspace(4) %1, ptr addrspace(4) %functor.ascast, align 8, !tbaa !17
  %call.i3 = tail call spir_func i64 @_Z12get_local_idj(i32 noundef 1) #7
  %idxprom = and i64 %call.i3, 4294967295
  %arrayidx = getelementptr inbounds nuw double, ptr addrspace(4) addrspacecast (ptr addrspace(3) @shared to ptr addrspace(4)), i64 %idxprom
  store double 0.000000e+00, ptr addrspace(4) %arrayidx, align 8, !tbaa !13
  %cmp = icmp eq i64 %idxprom, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store double 5.000000e+00, ptr addrspace(4) addrspacecast (ptr addrspace(3) @shared to ptr addrspace(4)), align 8, !tbaa !13
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %2 = ptrtoint ptr addrspace(1) %out.coerce to i64
  %3 = inttoptr i64 %2 to ptr addrspace(4)
  tail call spir_func void @_Z7barrierj(i32 noundef 1) #7
  %4 = load double, ptr addrspace(4) %arrayidx, align 8, !tbaa !13
  tail call spir_func void @_Z7barrierj(i32 noundef 1) #7
  call spir_func void @_Z19intraBlockReductionRK7ReducerdPdS2_(ptr addrspace(4) noundef align 8 dereferenceable(8) %functor.ascast, double noundef %4, ptr addrspace(4) noundef %3, ptr addrspace(4) noundef addrspacecast (ptr addrspace(3) @shared to ptr addrspace(4))) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %functor) #8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #6

attributes #0 = { convergent mustprogress noinline nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nofree nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent noinline nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { convergent nounwind }
attributes #8 = { nounwind }

!opencl.ocl.version = !{!0, !1}
!llvm.ident = !{!2}
!llvm.errno.tbaa = !{!3, !7}
!llvm.module.flags = !{!11, !12}

!0 = !{i32 0, i32 0}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 22.1.0 (git@github.com:CHIP-SPV/llvm-project.git fb0b9f488e26ce8a62f2900fc26d878181894509)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{i32 7, !"frame-pointer", i32 2}
!13 = !{!14, !14, i64 0}
!14 = !{!"double", !5, i64 0}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.mustprogress"}
!17 = !{!18, !19, i64 0}
!18 = !{!"_ZTS7Reducer", !19, i64 0}
!19 = !{!"p1 double", !20, i64 0}
!20 = !{!"any pointer", !5, i64 0}

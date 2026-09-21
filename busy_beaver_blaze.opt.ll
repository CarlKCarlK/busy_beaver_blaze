; ModuleID = '/home/carlk/programs/busy_beaver_blaze/busy_beaver_blaze.ll'
source_filename = "busy_beaver_blaze"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@llvm.used = appending global [1 x ptr] [ptr @horizontal_average], section "llvm.metadata"

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #0

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @horizontal_average(ptr nofree nonnull readonly captures(none) %v0, i64 %v1, i8 %v2, i64 %v3, ptr nofree writeonly captures(address_is_null) %v4, i64 %v5) #1 !dbg !4 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !8
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #5, !dbg !12
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5, !dbg !13
  %v9.i = zext nneg i32 %v2.i to i64, !dbg !14
  %v10.i = zext nneg i32 %v3.i to i64, !dbg !15
  %v17.i = mul nuw nsw i64 %v9.i, %v10.i, !dbg !16
  %v11.i = zext nneg i32 %v4.i to i64, !dbg !17
  %v18.i = add nuw nsw i64 %v17.i, %v11.i, !dbg !16
  %v4.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #5, !dbg !18
  %v6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #5, !dbg !18
  %v5.i = icmp eq i32 %v4.i2, 1, !dbg !18
  %v7.i = icmp eq i32 %v6.i, 1, !dbg !18
  %v8.not.not.i = and i1 %v5.i, %v7.i, !dbg !21
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #5, !dbg !21
  %v14.i = icmp eq i32 %v13.i, 1, !dbg !21
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #5, !dbg !21
  %v16.i = icmp eq i32 %v15.i, 1, !dbg !21
  %v17.i3 = and i1 %v14.i, %v16.i, !dbg !21
  %.v18.i = and i1 %v8.not.not.i, %v17.i3, !dbg !22
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1, !dbg !23
  %v17.not = icmp ult i64 %v22.i, %v5, !dbg !24
  br i1 %v17.not, label %bb3, label %bb17, !dbg !24

bb3:                                              ; preds = %entry
  %v19 = mul i64 %v3, %v22.i, !dbg !25
  %v22.not5.not = icmp eq i64 %v3, 0, !dbg !26
  br i1 %v22.not5.not, label %bb11, label %bb5.preheader, !dbg !26

bb5.preheader:                                    ; preds = %bb3
  %0 = tail call i64 @llvm.usub.sat.i64(i64 %v1, i64 %v19), !dbg !27
  %1 = add i64 %v3, -1, !dbg !27
  %.not.not = icmp ugt i64 %0, %1, !dbg !27
  br i1 %.not.not, label %bb5.preheader.split, label %bb24, !dbg !27

bb5.preheader.split:                              ; preds = %bb5.preheader
  %invariant.gep = getelementptr i8, ptr %v0, i64 %v19, !dbg !27
  %xtraiter = and i64 %v3, 3, !dbg !27
  %2 = icmp ult i64 %v3, 4, !dbg !27
  br i1 %2, label %bb5.epil.preheader, label %bb5.preheader.split.new, !dbg !27

bb5.preheader.split.new:                          ; preds = %bb5.preheader.split
  %unroll_iter = and i64 %v3, -4, !dbg !27
  br label %bb5, !dbg !27

bb5:                                              ; preds = %bb5, %bb5.preheader.split.new
  %v217 = phi i64 [ 0, %bb5.preheader.split.new ], [ %v34.3, %bb5 ]
  %v206 = phi i32 [ 0, %bb5.preheader.split.new ], [ %v33.3, %bb5 ]
  %niter = phi i64 [ 0, %bb5.preheader.split.new ], [ %niter.next.3, %bb5 ]
  %gep = getelementptr i8, ptr %invariant.gep, i64 %v217, !dbg !27
  %v29 = load i8, ptr %gep, align 1, !dbg !27
  %v30.not = icmp eq i8 %v29, %v2, !dbg !27
  %v32 = zext i1 %v30.not to i32, !dbg !27
  %v33 = add i32 %v206, %v32, !dbg !27
  %3 = getelementptr i8, ptr %invariant.gep, i64 %v217, !dbg !27
  %gep.1 = getelementptr i8, ptr %3, i64 1, !dbg !27
  %v29.1 = load i8, ptr %gep.1, align 1, !dbg !27
  %v30.not.1 = icmp eq i8 %v29.1, %v2, !dbg !27
  %v32.1 = zext i1 %v30.not.1 to i32, !dbg !27
  %v33.1 = add i32 %v33, %v32.1, !dbg !27
  %4 = getelementptr i8, ptr %invariant.gep, i64 %v217, !dbg !27
  %gep.2 = getelementptr i8, ptr %4, i64 2, !dbg !27
  %v29.2 = load i8, ptr %gep.2, align 1, !dbg !27
  %v30.not.2 = icmp eq i8 %v29.2, %v2, !dbg !27
  %v32.2 = zext i1 %v30.not.2 to i32, !dbg !27
  %v33.2 = add i32 %v33.1, %v32.2, !dbg !27
  %5 = getelementptr i8, ptr %invariant.gep, i64 %v217, !dbg !27
  %gep.3 = getelementptr i8, ptr %5, i64 3, !dbg !27
  %v29.3 = load i8, ptr %gep.3, align 1, !dbg !27
  %v30.not.3 = icmp eq i8 %v29.3, %v2, !dbg !27
  %v32.3 = zext i1 %v30.not.3 to i32, !dbg !27
  %v33.3 = add i32 %v33.2, %v32.3, !dbg !27
  %v34.3 = add nuw i64 %v217, 4, !dbg !28
  %niter.next.3 = add i64 %niter, 4, !dbg !26
  %niter.ncmp.3 = icmp eq i64 %niter.next.3, %unroll_iter, !dbg !26
  br i1 %niter.ncmp.3, label %bb11.loopexit.unr-lcssa, label %bb5, !dbg !26

bb11.loopexit.unr-lcssa:                          ; preds = %bb5
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0, !dbg !26
  br i1 %lcmp.mod.not, label %bb11.loopexit, label %bb5.epil.preheader, !dbg !26

bb5.epil.preheader:                               ; preds = %bb11.loopexit.unr-lcssa, %bb5.preheader.split
  %v217.epil.init = phi i64 [ 0, %bb5.preheader.split ], [ %v34.3, %bb11.loopexit.unr-lcssa ]
  %v206.epil.init = phi i32 [ 0, %bb5.preheader.split ], [ %v33.3, %bb11.loopexit.unr-lcssa ]
  %lcmp.mod9 = icmp ne i64 %xtraiter, 0, !dbg !26
  tail call void @llvm.assume(i1 %lcmp.mod9), !dbg !26
  br label %bb5.epil, !dbg !26

bb5.epil:                                         ; preds = %bb5.epil, %bb5.epil.preheader
  %v217.epil = phi i64 [ %v34.epil, %bb5.epil ], [ %v217.epil.init, %bb5.epil.preheader ]
  %v206.epil = phi i32 [ %v33.epil, %bb5.epil ], [ %v206.epil.init, %bb5.epil.preheader ]
  %epil.iter = phi i64 [ %epil.iter.next, %bb5.epil ], [ 0, %bb5.epil.preheader ]
  %gep.epil = getelementptr i8, ptr %invariant.gep, i64 %v217.epil, !dbg !27
  %v29.epil = load i8, ptr %gep.epil, align 1, !dbg !27
  %v30.not.epil = icmp eq i8 %v29.epil, %v2, !dbg !27
  %v32.epil = zext i1 %v30.not.epil to i32, !dbg !27
  %v33.epil = add i32 %v206.epil, %v32.epil, !dbg !27
  %v34.epil = add nuw i64 %v217.epil, 1, !dbg !28
  %epil.iter.next = add i64 %epil.iter, 1, !dbg !26
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter, !dbg !26
  br i1 %epil.iter.cmp.not, label %bb11.loopexit, label %bb5.epil, !dbg !26, !llvm.loop !29

bb11.loopexit:                                    ; preds = %bb5.epil, %bb11.loopexit.unr-lcssa
  %v33.lcssa = phi i32 [ %v33.3, %bb11.loopexit.unr-lcssa ], [ %v33.epil, %bb5.epil ], !dbg !27
  %6 = mul i32 %v33.lcssa, 255
  br label %bb11, !dbg !31

bb11:                                             ; preds = %bb11.loopexit, %bb3
  %v20.lcssa = phi i32 [ 0, %bb3 ], [ %6, %bb11.loopexit ]
  %v36 = trunc i64 %v3 to i32, !dbg !31
  %v37.not = icmp eq i32 %v36, 0, !dbg !32
  br i1 %v37.not, label %bb25, label %bb13, !dbg !32

bb13:                                             ; preds = %bb11
  %v571.not = icmp eq ptr %v4, null
  br i1 %v571.not, label %bb17, label %bb14, !dbg !33

bb14:                                             ; preds = %bb13
  %v46 = getelementptr inbounds i8, ptr %v4, i64 %v22.i
  %v39 = udiv i32 %v20.lcssa, %v36, !dbg !32
  %v40 = trunc i32 %v39 to i8, !dbg !32
  store i8 %v40, ptr %v46, align 1, !dbg !34
  br label %bb17, !dbg !35

bb17:                                             ; preds = %bb13, %bb14, %entry
  ret void, !dbg !36

bb24:                                             ; preds = %bb5.preheader
  tail call void @llvm.trap() #5, !dbg !27
  unreachable, !dbg !27

bb25:                                             ; preds = %bb11
  tail call void @llvm.trap() #5, !dbg !32
  unreachable, !dbg !32
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65) i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.usub.sat.i64(i64, i64) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #4

attributes #0 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #1 = { convergent nounwind memory(argmem: readwrite, inaccessiblemem: write) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #5 = { convergent }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !1, producer: "cuda-oxide", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "gpu_oxide.rs", directory: "src")
!2 = !{i32 2, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "horizontal_average", linkageName: "horizontal_average", scope: !1, file: !1, line: 21, type: !5, scopeLine: 21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{}
!8 = !DILocation(line: 652, column: 13, scope: !9, inlinedAt: !11)
!9 = distinct !DISubprogram(name: "cuda_device::__internal::index_1d::<cuda_device::__internal::UnknownDomain, cuda_device::__internal::NativeCoordinates>", linkageName: "_RINvNtNtCsfEVodaNHerO_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECshRFRwAchouq_17busy_beaver_blaze", scope: !10, file: !10, line: 648, type: !5, scopeLine: 648, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!10 = !DIFile(filename: "thread.rs", directory: "/home/carlk/.cargo/git/checkouts/cuda-oxide-6d394bb007f5e114/b0f961d/crates/cuda-device/src")
!11 = distinct !DILocation(line: 27, column: 27, scope: !4)
!12 = !DILocation(line: 653, column: 13, scope: !9, inlinedAt: !11)
!13 = !DILocation(line: 654, column: 13, scope: !9, inlinedAt: !11)
!14 = !DILocation(line: 602, column: 21, scope: !9, inlinedAt: !11)
!15 = !DILocation(line: 603, column: 26, scope: !9, inlinedAt: !11)
!16 = !DILocation(line: 608, column: 13, scope: !9, inlinedAt: !11)
!17 = !DILocation(line: 604, column: 22, scope: !9, inlinedAt: !11)
!18 = !DILocation(line: 619, column: 13, scope: !19, inlinedAt: !20)
!19 = distinct !DISubprogram(name: "cuda_device::__internal::one_dimensional_launch::<cuda_device::__internal::UnknownDomain, cuda_device::__internal::NativeCoordinates>", linkageName: "_RINvNtNtCsfEVodaNHerO_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECshRFRwAchouq_17busy_beaver_blaze", scope: !10, file: !10, line: 613, type: !5, scopeLine: 613, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!20 = distinct !DILocation(line: 656, column: 21, scope: !9, inlinedAt: !11)
!21 = !DILocation(line: 624, column: 13, scope: !19, inlinedAt: !20)
!22 = !DILocation(line: 626, column: 9, scope: !19, inlinedAt: !20)
!23 = !DILocation(line: 312, column: 21, scope: !9, inlinedAt: !11)
!24 = !DILocation(line: 29, column: 12, scope: !4)
!25 = !DILocation(line: 33, column: 28, scope: !4)
!26 = !DILocation(line: 36, column: 15, scope: !4)
!27 = !DILocation(line: 37, column: 16, scope: !4)
!28 = !DILocation(line: 40, column: 13, scope: !4)
!29 = distinct !{!29, !30}
!30 = !{!"llvm.loop.unroll.disable"}
!31 = !DILocation(line: 42, column: 45, scope: !4)
!32 = !DILocation(line: 42, column: 21, scope: !4)
!33 = !DILocation(line: 43, column: 16, scope: !4)
!34 = !DILocation(line: 44, column: 13, scope: !4)
!35 = !DILocation(line: 43, column: 9, scope: !4)
!36 = !DILocation(line: 46, column: 6, scope: !4)

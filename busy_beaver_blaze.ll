; ModuleID = 'builtin.module'
source_filename = "busy_beaver_blaze"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare void @llvm.trap()

define ptx_kernel void @horizontal_average(ptr nonnull %v0, i64 %v1, i8 %v2, i64 %v3, ptr %v4, i64 %v5) #0 !dbg !3 {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v5, 1
  br label %bb0
bb0:
  %v10 = phi { ptr, i64 } [ %v7, %entry ]
  %v11 = phi i8 [ %v2, %entry ]
  %v12 = phi i64 [ %v3, %entry ]
  %v13 = phi { ptr, i64 } [ %v9, %entry ]
  %v14 = alloca {  }, align 1
  %v15 = call i64 @_RINvNtNtCsfEVodaNHerO_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECshRFRwAchouq_17busy_beaver_blaze(ptr %v14) #0, !dbg !4
  br label %bb1, !dbg !4
bb1:
  %v16 = extractvalue { ptr, i64 } %v13, 1, !dbg !7
  %v17 = icmp uge i64 %v15, %v16, !dbg !8
  %v18 = xor i1 %v17, 1, !dbg !8
  br i1 %v18, label %bb3, label %bb2, !dbg !8
bb2:
  br label %bb17, !dbg !9
bb3:
  %v19 = mul i64 %v15, %v12, !dbg !10
  br label %bb4, !dbg !11
bb4:
  %v20 = phi i32 [ 0, %bb3 ], [ %v33, %bb10 ]
  %v21 = phi i64 [ 0, %bb3 ], [ %v34, %bb10 ]
  %v22 = icmp ult i64 %v21, %v12, !dbg !12
  %v23 = xor i1 %v22, 1, !dbg !12
  br i1 %v23, label %bb11, label %bb5, !dbg !12
bb5:
  %v24 = add i64 %v19, %v21, !dbg !13
  %v25 = extractvalue { ptr, i64 } %v10, 1, !dbg !14
  %v26 = icmp ult i64 %v24, %v25, !dbg !14
  br i1 %v26, label %bb6, label %bb24, !dbg !14
bb6:
  br label %bb7, !dbg !14
bb7:
  %v27 = extractvalue { ptr, i64 } %v10, 0, !dbg !14
  %v28 = getelementptr inbounds i8, ptr %v27, i64 %v24, !dbg !14
  %v29 = load i8, ptr %v28, align 1, !dbg !14
  %v30 = icmp eq i8 %v29, %v11, !dbg !14
  %v31 = xor i1 %v30, 1, !dbg !14
  br i1 %v31, label %bb9, label %bb8, !dbg !14
bb8:
  %v32 = add i32 %v20, 1, !dbg !15
  br label %bb10, !dbg !16
bb9:
  br label %bb10, !dbg !16
bb10:
  %v33 = phi i32 [ %v32, %bb8 ], [ %v20, %bb9 ]
  %v34 = add i64 %v21, 1, !dbg !17
  br label %bb4, !dbg !11
bb11:
  %v35 = mul i32 %v20, 255, !dbg !18
  %v36 = trunc i64 %v12 to i32, !dbg !19
  %v37 = icmp eq i32 %v36, 0, !dbg !20
  %v38 = xor i1 %v37, 1, !dbg !20
  br i1 %v38, label %bb12, label %bb25, !dbg !20
bb12:
  br label %bb13, !dbg !20
bb13:
  %v39 = udiv i32 %v35, %v36, !dbg !20
  %v40 = trunc i32 %v39 to i8, !dbg !20
  %v41 = icmp eq i64 %v15, 18446744073709551615, !dbg !21
  br i1 %v41, label %bb20, label %bb18, !dbg !21
bb14:
  %v42 = extractvalue { ptr } %v53, 0
  store i8 %v40, ptr %v42, align 1, !dbg !22
  br label %bb16, !dbg !23
bb15:
  br label %bb16, !dbg !23
bb16:
  br label %bb17, !dbg !9
bb17:
  ret void, !dbg !9
bb18:
  %v43 = icmp ult i64 %v15, %v16, !dbg !24
  %v44 = xor i1 %v43, 1, !dbg !24
  br i1 %v44, label %bb20, label %bb19, !dbg !24
bb19:
  %v45 = extractvalue { ptr, i64 } %v13, 0, !dbg !25
  %v46 = getelementptr inbounds i8, ptr %v45, i64 %v15, !dbg !28
  %v47 = insertvalue { ptr } undef, ptr %v46, 0, !dbg !29
  %v48 = extractvalue { ptr } %v47, 0
  br label %bb21, !dbg !30
bb20:
  %v49 = inttoptr i64 0 to ptr
  %v50 = insertvalue { ptr } undef, ptr %v49, 0, !dbg !31
  %v51 = extractvalue { ptr } %v50, 0
  br label %bb21, !dbg !30
bb21:
  %v52 = phi ptr [ %v48, %bb19 ], [ %v51, %bb20 ]
  %v53 = insertvalue { ptr } undef, ptr %v52, 0
  %v54 = extractvalue { ptr } %v53, 0
  %v55 = ptrtoint ptr %v54 to i64
  %v56 = sub i64 %v55, 0
  %v57 = icmp ule i64 %v56, 0
  %v58 = add i64 %v56, 0
  %v59 = select i1 %v57, i64 %v58, i64 1
  %v60 = icmp eq i64 %v59, 1, !dbg !32
  br i1 %v60, label %bb14, label %bb22, !dbg !32
bb22:
  %v61 = icmp eq i64 %v59, 0, !dbg !32
  br i1 %v61, label %bb15, label %bb23, !dbg !32
bb23:
  unreachable, !dbg !33
bb24:
  call void @llvm.trap() #0, !dbg !14
  unreachable, !dbg !14
bb25:
  call void @llvm.trap() #0, !dbg !20
  unreachable, !dbg !20
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define i64 @_RINvNtNtCsfEVodaNHerO_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECshRFRwAchouq_17busy_beaver_blaze(ptr %v0) alwaysinline #0 !dbg !35 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0, !dbg !36
  br label %bb1, !dbg !36
bb1:
  %v3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0, !dbg !37
  br label %bb2, !dbg !37
bb2:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0, !dbg !38
  br label %bb7, !dbg !38
bb3:
  %v5 = xor i1 %v20, 1, !dbg !39
  br i1 %v5, label %bb5, label %bb4, !dbg !39
bb4:
  %v6 = icmp ne i64 %v19, 18446744073709551615, !dbg !40
  br label %bb6, !dbg !39
bb5:
  br label %bb6, !dbg !39
bb6:
  %v7 = phi i1 [ %v6, %bb4 ], [ 0, %bb5 ]
  %v8 = xor i1 %v7, 1, !dbg !41
  br i1 %v8, label %bb14, label %bb13, !dbg !41
bb7:
  %v9 = zext i32 %v2 to i64, !dbg !42
  %v10 = zext i32 %v3 to i64, !dbg !43
  %v11 = zext i32 %v4 to i64, !dbg !44
  %v12 = icmp eq i64 %v10, 0, !dbg !45
  br i1 %v12, label %bb10, label %bb8, !dbg !45
bb8:
  %v13 = sub i64 18446744073709551615, %v11, !dbg !46
  %v14 = udiv i64 %v13, %v10, !dbg !46
  %v15 = icmp ugt i64 %v9, %v14, !dbg !47
  %v16 = xor i1 %v15, 1, !dbg !47
  br i1 %v16, label %bb11, label %bb9, !dbg !47
bb9:
  br label %bb10, !dbg !48
bb10:
  br label %bb12, !dbg !49
bb11:
  %v17 = mul i64 %v9, %v10, !dbg !50
  %v18 = add i64 %v17, %v11, !dbg !50
  br label %bb12, !dbg !49
bb12:
  %v19 = phi i64 [ 18446744073709551615, %bb10 ], [ %v18, %bb11 ]
  %v20 = call i1 @_RINvNtNtCsfEVodaNHerO_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECshRFRwAchouq_17busy_beaver_blaze(ptr %v1) #0, !dbg !39
  br label %bb3, !dbg !39
bb13:
  %v21 = icmp eq i64 %v19, 18446744073709551615, !dbg !51
  br i1 %v21, label %bb14, label %bb15, !dbg !51
bb14:
  br label %bb15, !dbg !52
bb15:
  %v22 = phi i64 [ %v19, %bb13 ], [ 18446744073709551615, %bb14 ]
  ret i64 %v22, !dbg !53
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()

define i1 @_RINvNtNtCsfEVodaNHerO_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECshRFRwAchouq_17busy_beaver_blaze(ptr %v0) alwaysinline #0 !dbg !54 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = icmp eq i8 0, 1, !dbg !55
  %v3 = xor i1 %v2, 1, !dbg !55
  br i1 %v3, label %bb2, label %bb1, !dbg !55
bb1:
  br label %bb8, !dbg !56
bb2:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0, !dbg !57
  br label %bb3, !dbg !57
bb3:
  %v5 = icmp eq i32 %v4, 1, !dbg !57
  br i1 %v5, label %bb4, label %bb5, !dbg !57
bb4:
  %v6 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #0, !dbg !58
  br label %bb6, !dbg !58
bb5:
  br label %bb7, !dbg !57
bb6:
  %v7 = icmp eq i32 %v6, 1, !dbg !58
  br label %bb7, !dbg !57
bb7:
  %v8 = phi i1 [ 0, %bb5 ], [ %v7, %bb6 ]
  br label %bb8, !dbg !56
bb8:
  %v9 = phi i1 [ 1, %bb1 ], [ %v8, %bb7 ]
  %v10 = xor i1 %v2, 1, !dbg !59
  br i1 %v10, label %bb9, label %bb10, !dbg !59
bb9:
  %v11 = icmp eq i8 0, 2, !dbg !60
  %v12 = xor i1 %v11, 1, !dbg !60
  br i1 %v12, label %bb11, label %bb10, !dbg !60
bb10:
  br label %bb17, !dbg !61
bb11:
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0, !dbg !62
  br label %bb12, !dbg !62
bb12:
  %v14 = icmp eq i32 %v13, 1, !dbg !62
  br i1 %v14, label %bb13, label %bb14, !dbg !62
bb13:
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0, !dbg !63
  br label %bb15, !dbg !63
bb14:
  br label %bb16, !dbg !62
bb15:
  %v16 = icmp eq i32 %v15, 1, !dbg !63
  br label %bb16, !dbg !62
bb16:
  %v17 = phi i1 [ 0, %bb14 ], [ %v16, %bb15 ]
  br label %bb17, !dbg !61
bb17:
  %v18 = phi i1 [ 1, %bb10 ], [ %v17, %bb16 ]
  %v19 = xor i1 %v9, 1, !dbg !64
  br i1 %v19, label %bb19, label %bb18, !dbg !64
bb18:
  br label %bb20, !dbg !64
bb19:
  br label %bb20, !dbg !64
bb20:
  %v20 = phi i1 [ %v18, %bb18 ], [ 0, %bb19 ]
  ret i1 %v20, !dbg !65
}


@llvm.used = appending global [1 x ptr] [ptr @horizontal_average], section "llvm.metadata"

attributes #0 = { convergent }

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!66, !67}
!66 = !{i32 2, !"Dwarf Version", i32 2}
!67 = !{i32 2, !"Debug Info Version", i32 3}
!0 = !DIFile(filename: "gpu_oxide.rs", directory: "src")
!1 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !0, producer: "cuda-oxide", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!2 = !DISubroutineType(types: !{null})
!3 = distinct !DISubprogram(name: "horizontal_average", linkageName: "horizontal_average", scope: !0, file: !0, line: 21, type: !2, scopeLine: 21, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!4 = !DILocation(line: 27, column: 27, scope: !3)
!5 = !DIFile(filename: "disjoint.rs", directory: "/home/carlk/.cargo/git/checkouts/cuda-oxide-6d394bb007f5e114/b0f961d/crates/cuda-device/src")
!6 = !DILexicalBlockFile(scope: !3, file: !5, discriminator: 0)
!7 = !DILocation(line: 381, column: 9, scope: !6)
!8 = !DILocation(line: 29, column: 12, scope: !3)
!9 = !DILocation(line: 46, column: 6, scope: !3)
!10 = !DILocation(line: 33, column: 28, scope: !3)
!11 = !DILocation(line: 36, column: 9, scope: !3)
!12 = !DILocation(line: 36, column: 15, scope: !3)
!13 = !DILocation(line: 37, column: 22, scope: !3)
!14 = !DILocation(line: 37, column: 16, scope: !3)
!15 = !DILocation(line: 38, column: 17, scope: !3)
!16 = !DILocation(line: 37, column: 13, scope: !3)
!17 = !DILocation(line: 40, column: 13, scope: !3)
!18 = !DILocation(line: 42, column: 22, scope: !3)
!19 = !DILocation(line: 42, column: 45, scope: !3)
!20 = !DILocation(line: 42, column: 21, scope: !3)
!21 = !DILocation(line: 489, column: 35, scope: !6)
!22 = !DILocation(line: 44, column: 13, scope: !3)
!23 = !DILocation(line: 43, column: 9, scope: !3)
!24 = !DILocation(line: 489, column: 53, scope: !6)
!25 = !DILocation(line: 496, column: 33, scope: !6)
!26 = !DIFile(filename: "mut_ptr.rs", directory: "/rustc/e457a7b0d326d67b4322ef0d11bd715cfaeda48f/library/core/src/ptr")
!27 = !DILexicalBlockFile(scope: !3, file: !26, discriminator: 0)
!28 = !DILocation(line: 971, column: 18, scope: !27)
!29 = !DILocation(line: 496, column: 13, scope: !6)
!30 = !DILocation(line: 489, column: 9, scope: !6)
!31 = !DILocation(line: 498, column: 13, scope: !6)
!32 = !DILocation(line: 43, column: 16, scope: !3)
!33 = !DILocation(line: 21, column: 5, scope: !3)
!34 = !DIFile(filename: "thread.rs", directory: "/home/carlk/.cargo/git/checkouts/cuda-oxide-6d394bb007f5e114/b0f961d/crates/cuda-device/src")
!35 = distinct !DISubprogram(name: "cuda_device::__internal::index_1d::<cuda_device::__internal::UnknownDomain, cuda_device::__internal::NativeCoordinates>", linkageName: "_RINvNtNtCsfEVodaNHerO_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECshRFRwAchouq_17busy_beaver_blaze", scope: !34, file: !34, line: 648, type: !2, scopeLine: 648, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!36 = !DILocation(line: 652, column: 13, scope: !35)
!37 = !DILocation(line: 653, column: 13, scope: !35)
!38 = !DILocation(line: 654, column: 13, scope: !35)
!39 = !DILocation(line: 656, column: 21, scope: !35)
!40 = !DILocation(line: 656, column: 54, scope: !35)
!41 = !DILocation(line: 312, column: 21, scope: !35)
!42 = !DILocation(line: 602, column: 21, scope: !35)
!43 = !DILocation(line: 603, column: 26, scope: !35)
!44 = !DILocation(line: 604, column: 22, scope: !35)
!45 = !DILocation(line: 605, column: 12, scope: !35)
!46 = !DILocation(line: 605, column: 39, scope: !35)
!47 = !DILocation(line: 605, column: 31, scope: !35)
!48 = !DILocation(line: 605, column: 74, scope: !35)
!49 = !DILocation(line: 605, column: 9, scope: !35)
!50 = !DILocation(line: 608, column: 13, scope: !35)
!51 = !DILocation(line: 312, column: 30, scope: !35)
!52 = !DILocation(line: 312, column: 18, scope: !35)
!53 = !DILocation(line: 658, column: 6, scope: !35)
!54 = distinct !DISubprogram(name: "cuda_device::__internal::one_dimensional_launch::<cuda_device::__internal::UnknownDomain, cuda_device::__internal::NativeCoordinates>", linkageName: "_RINvNtNtCsfEVodaNHerO_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECshRFRwAchouq_17busy_beaver_blaze", scope: !34, file: !34, line: 613, type: !2, scopeLine: 613, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!55 = !DILocation(line: 616, column: 28, scope: !54)
!56 = !DILocation(line: 616, column: 25, scope: !54)
!57 = !DILocation(line: 619, column: 13, scope: !54)
!58 = !DILocation(line: 619, column: 41, scope: !54)
!59 = !DILocation(line: 621, column: 28, scope: !54)
!60 = !DILocation(line: 621, column: 59, scope: !54)
!61 = !DILocation(line: 621, column: 25, scope: !54)
!62 = !DILocation(line: 624, column: 13, scope: !54)
!63 = !DILocation(line: 624, column: 41, scope: !54)
!64 = !DILocation(line: 626, column: 9, scope: !54)
!65 = !DILocation(line: 627, column: 6, scope: !54)

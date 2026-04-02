; ModuleID = 'd1'
source_filename = "src/intrinsic_wrappers.ll"

%struct.int2 = type { i32, i32 }
%struct.int4 = type { i32, i32, i32, i32 }

; Function Attrs: nounwind memory(read)
; Unknown intrinsic
declare i32 @llvm.nvvm.load.matrix.n1.p3i32(i32, ptr addrspace(3) captures(none)) #0

; Function Attrs: nounwind memory(read)
; Unknown intrinsic
declare <2 x i32> @llvm.nvvm.load.matrix.n2.p3i32(i32, ptr addrspace(3) captures(none)) #0

; Function Attrs: nounwind memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.nvvm.load.matrix.n4.p3i32(i32, ptr addrspace(3) captures(none)) #0

; Function Attrs: alwaysinline inlinehint nounwind
define i32 @__nvvm_ldsm_b16_m8n8_n1(ptr %src) #1 {
  %ptr = addrspacecast ptr %src to ptr addrspace(3)
  %res = tail call i32 @llvm.nvvm.load.matrix.n1.p3i32(i32 0, ptr addrspace(3) %ptr)
  ret i32 %res
}

; Function Attrs: alwaysinline inlinehint nounwind
define i32 @__nvvm_ldsm_b16_m8n8_t1(ptr %src) #1 {
  %ptr = addrspacecast ptr %src to ptr addrspace(3)
  %res = tail call i32 @llvm.nvvm.load.matrix.n1.p3i32(i32 1, ptr addrspace(3) %ptr)
  ret i32 %res
}

; Function Attrs: alwaysinline inlinehint nounwind
define %struct.int2 @__nvvm_ldsm_b16_m8n8_n2(ptr %src) #1 {
  %ptr = addrspacecast ptr %src to ptr addrspace(3)
  %vec = tail call <2 x i32> @llvm.nvvm.load.matrix.n2.p3i32(i32 0, ptr addrspace(3) %ptr)
  %e0 = extractelement <2 x i32> %vec, i32 0
  %e1 = extractelement <2 x i32> %vec, i32 1
  %res0 = insertvalue %struct.int2 undef, i32 %e0, 0
  %res1 = insertvalue %struct.int2 %res0, i32 %e1, 1
  ret %struct.int2 %res1
}

; Function Attrs: alwaysinline inlinehint nounwind
define %struct.int2 @__nvvm_ldsm_b16_m8n8_t2(ptr %src) #1 {
  %ptr = addrspacecast ptr %src to ptr addrspace(3)
  %vec = tail call <2 x i32> @llvm.nvvm.load.matrix.n2.p3i32(i32 1, ptr addrspace(3) %ptr)
  %e0 = extractelement <2 x i32> %vec, i32 0
  %e1 = extractelement <2 x i32> %vec, i32 1
  %res0 = insertvalue %struct.int2 undef, i32 %e0, 0
  %res1 = insertvalue %struct.int2 %res0, i32 %e1, 1
  ret %struct.int2 %res1
}

; Function Attrs: alwaysinline inlinehint nounwind
define %struct.int4 @__nvvm_ldsm_b16_m8n8_n4(ptr %src) #1 {
  %ptr = addrspacecast ptr %src to ptr addrspace(3)
  %vec = tail call <4 x i32> @llvm.nvvm.load.matrix.n4.p3i32(i32 0, ptr addrspace(3) %ptr)
  %e0 = extractelement <4 x i32> %vec, i32 0
  %e1 = extractelement <4 x i32> %vec, i32 1
  %e2 = extractelement <4 x i32> %vec, i32 2
  %e3 = extractelement <4 x i32> %vec, i32 3
  %res0 = insertvalue %struct.int4 undef, i32 %e0, 0
  %res1 = insertvalue %struct.int4 %res0, i32 %e1, 1
  %res2 = insertvalue %struct.int4 %res1, i32 %e2, 2
  %res3 = insertvalue %struct.int4 %res2, i32 %e3, 3
  ret %struct.int4 %res3
}

; Function Attrs: alwaysinline inlinehint nounwind
define %struct.int4 @__nvvm_ldsm_b16_m8n8_t4(ptr %src) #1 {
  %ptr = addrspacecast ptr %src to ptr addrspace(3)
  %vec = tail call <4 x i32> @llvm.nvvm.load.matrix.n4.p3i32(i32 1, ptr addrspace(3) %ptr)
  %e0 = extractelement <4 x i32> %vec, i32 0
  %e1 = extractelement <4 x i32> %vec, i32 1
  %e2 = extractelement <4 x i32> %vec, i32 2
  %e3 = extractelement <4 x i32> %vec, i32 3
  %res0 = insertvalue %struct.int4 undef, i32 %e0, 0
  %res1 = insertvalue %struct.int4 %res0, i32 %e1, 1
  %res2 = insertvalue %struct.int4 %res1, i32 %e2, 2
  %res3 = insertvalue %struct.int4 %res2, i32 %e3, 3
  ret %struct.int4 %res3
}

; Function Attrs: alwaysinline inlinehint nounwind
define i32 @__nvvm_get_smem_pointer(ptr %ptr) #1 {
  %smem_ptr = addrspacecast ptr %ptr to ptr addrspace(3)
  %int = ptrtoint ptr addrspace(3) %smem_ptr to i32
  ret i32 %int
}

attributes #0 = { nounwind memory(read) }
attributes #1 = { alwaysinline inlinehint nounwind }

!nvvmir.version = !{!0}

!0 = !{i32 2, i32 0, i32 3, i32 1}

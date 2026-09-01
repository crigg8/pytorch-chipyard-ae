; RUN: buddy-llc -mtriple=riscv64 -mattr=+v -verify-machineinstrs %s -o - | FileCheck %s

declare <vscale x 4 x i32> @llvm.riscv.vle.nxv4i32(<vscale x 4 x i32>, ptr, i64)
declare <vscale x 4 x i32> @llvm.riscv.vlse.nxv4i32(<vscale x 4 x i32>, ptr, i64, i64)
declare void @llvm.riscv.vse.nxv4i32(<vscale x 4 x i32>, ptr, i64)

define i32 @vle_then_lw(ptr align 4 %vecbase, ptr align 4 %scalbase, i64 %vecoff, i64 %vl) nounwind {
; CHECK-LABEL: vle_then_lw:
; CHECK:       add [[VBASE:[a-z0-9]+]], a0, a2
; CHECK:       vle32.v v{{[0-9]+}}, ([[VBASE]])
; CHECK-NOT:   fence
; CHECK-NOT:   vle32.v {{.*}}\(t0\)
; CHECK:       addi t0, a1, 2047
; CHECK:       lw {{[a-z0-9]+}}, 1(t0)
entry:
  %vecaddr = getelementptr i8, ptr %vecbase, i64 %vecoff
  %vec = call <vscale x 4 x i32> @llvm.riscv.vle.nxv4i32(<vscale x 4 x i32> undef, ptr %vecaddr, i64 %vl)
  %mid = getelementptr i8, ptr %scalbase, i64 2047
  %final = getelementptr i8, ptr %mid, i64 1
  %s = load i32, ptr %final, align 4
  %elt = extractelement <vscale x 4 x i32> %vec, i64 0
  %sum = add i32 %elt, %s
  ret i32 %sum
}

define i32 @vlse_then_lw(ptr align 4 %vecbase, ptr align 4 %scalbase, i64 %vecoff, i64 %stride, i64 %vl) nounwind {
; CHECK-LABEL: vlse_then_lw:
; CHECK:       add [[VBASE:[a-z0-9]+]], a0, a2
; CHECK:       vlse32.v v{{[0-9]+}}, ([[VBASE]]), a3
; CHECK-NOT:   vlse32.v {{.*}}\(t0\)
; CHECK:       addi t0, a1, 2047
; CHECK:       lw {{[a-z0-9]+}}, 1(t0)
entry:
  %vecaddr = getelementptr i8, ptr %vecbase, i64 %vecoff
  %vec = call <vscale x 4 x i32> @llvm.riscv.vlse.nxv4i32(<vscale x 4 x i32> undef, ptr %vecaddr, i64 %stride, i64 %vl)
  %mid = getelementptr i8, ptr %scalbase, i64 2047
  %final = getelementptr i8, ptr %mid, i64 1
  %s = load i32, ptr %final, align 4
  %elt = extractelement <vscale x 4 x i32> %vec, i64 0
  %sum = add i32 %elt, %s
  ret i32 %sum
}

define void @vse_then_sw(ptr align 4 %srcvecbase, ptr align 4 %vecbase, ptr align 4 %scalbase, i64 %vecoff, i32 %val, i64 %vl) nounwind {
; CHECK-LABEL: vse_then_sw:
; CHECK:       add [[VBASE:[a-z0-9]+]], a1, a3
; CHECK:       addi [[SBASE:[a-z0-9]+]], a2, 2047
; CHECK:       vse32.v v{{[0-9]+}}, ([[VBASE]])
; CHECK-NEXT:  fence rw, rw
; CHECK-NOT:   sw a4, 1([[VBASE]])
; CHECK:       sw a4, 1([[SBASE]])
entry:
  %vec = call <vscale x 4 x i32> @llvm.riscv.vle.nxv4i32(<vscale x 4 x i32> undef, ptr %srcvecbase, i64 %vl)
  %vecaddr = getelementptr i8, ptr %vecbase, i64 %vecoff
  call void @llvm.riscv.vse.nxv4i32(<vscale x 4 x i32> %vec, ptr %vecaddr, i64 %vl)
  %mid = getelementptr i8, ptr %scalbase, i64 2047
  %final = getelementptr i8, ptr %mid, i64 1
  store i32 %val, ptr %final, align 4
  ret void
}

define i32 @vle_direct_base_then_lw(ptr align 4 %vecbase, ptr align 4 %scalbase, i64 %vl) nounwind {
; CHECK-LABEL: vle_direct_base_then_lw:
; CHECK:       vle32.v v{{[0-9]+}}, (a0)
; CHECK-NOT:   addi t0, a1, 2047
entry:
  %vec = call <vscale x 4 x i32> @llvm.riscv.vle.nxv4i32(<vscale x 4 x i32> undef, ptr %vecbase, i64 %vl)
  %mid = getelementptr i8, ptr %scalbase, i64 2047
  %final = getelementptr i8, ptr %mid, i64 1
  %s = load i32, ptr %final, align 4
  %elt = extractelement <vscale x 4 x i32> %vec, i64 0
  %sum = add i32 %elt, %s
  ret i32 %sum
}

define i64 @vle_then_nonmem_add(ptr %vecbase, ptr %scalbase, i64 %vecoff, i64 %vl) nounwind {
; CHECK-LABEL: vle_then_nonmem_add:
; CHECK-NOT:   addi t0, a1, 2047
entry:
  %vecaddr = getelementptr i8, ptr %vecbase, i64 %vecoff
  %vec = call <vscale x 4 x i32> @llvm.riscv.vle.nxv4i32(<vscale x 4 x i32> undef, ptr %vecaddr, i64 %vl)
  %mid = getelementptr i8, ptr %scalbase, i64 2047
  %pi = ptrtoint ptr %mid to i64
  %sum1 = add i64 %pi, 1
  %elt = extractelement <vscale x 4 x i32> %vec, i64 0
  %elt64 = zext i32 %elt to i64
  %sum = add i64 %sum1, %elt64
  ret i64 %sum
}

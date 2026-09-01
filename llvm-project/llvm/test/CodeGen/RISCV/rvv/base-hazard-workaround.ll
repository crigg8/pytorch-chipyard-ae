; RUN: llc -mtriple=riscv64 -mattr=+v -verify-machineinstrs %s | FileCheck %s

declare <vscale x 4 x i32> @llvm.riscv.vle.nxv4i32(<vscale x 4 x i32>, ptr, i64)
declare <vscale x 4 x i32> @llvm.riscv.vlse.nxv4i32(<vscale x 4 x i32>, ptr, i64, i64)
declare void @llvm.riscv.vse.nxv4i32(<vscale x 4 x i32>, ptr, i64)

define i32 @vle_then_lw(ptr align 4 %vecbase, ptr align 4 %scalbase, i64 %vecoff, i64 %vl) nounwind {
; CHECK-LABEL: vle_then_lw:
; CHECK:       add [[VBASE:[a-z0-9]+]], a0, a2
; CHECK:       vle32.v v{{[0-9]+}}, ([[VBASE]])
; RVV loads keep their base live until the vector consumer; only stores drain.
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
; CHECK:       addi t0, a2, 2047
; CHECK:       vse32.v v{{[0-9]+}}, ([[VBASE]])
; CHECK-NEXT:  fence rw, rw
; CHECK-NOT:   vse32.v {{.*}}\(t0\)
; CHECK:       sw a4, 1(t0)
entry:
  %vec = call <vscale x 4 x i32> @llvm.riscv.vle.nxv4i32(<vscale x 4 x i32> undef, ptr %srcvecbase, i64 %vl)
  %vecaddr = getelementptr i8, ptr %vecbase, i64 %vecoff
  call void @llvm.riscv.vse.nxv4i32(<vscale x 4 x i32> %vec, ptr %vecaddr, i64 %vl)
  %mid = getelementptr i8, ptr %scalbase, i64 2047
  %final = getelementptr i8, ptr %mid, i64 1
  store i32 %val, ptr %final, align 4
  ret void
}

define <vscale x 4 x i32> @vle_then_vle(ptr align 4 %vecbase0, ptr align 4 %vecbase1, i64 %vecoff, i64 %vl) nounwind {
; CHECK-LABEL: vle_then_vle:
; CHECK-DAG:   add [[VBASE:[as][0-9]+|t[1-6]|zero|ra|sp|gp|tp|fp]], a0, a2
; CHECK-DAG:   add t0, a1, a2
; CHECK:       vle32.v [[VEC0:v[0-9]+]], ([[VBASE]])
; CHECK:       vle32.v [[VEC1:v[0-9]+]], (t0)
entry:
  %vecaddr0 = getelementptr i8, ptr %vecbase0, i64 %vecoff
  %vec0 = call <vscale x 4 x i32> @llvm.riscv.vle.nxv4i32(<vscale x 4 x i32> undef, ptr %vecaddr0, i64 %vl)
  %vecaddr1 = getelementptr i8, ptr %vecbase1, i64 %vecoff
  %vec1 = call <vscale x 4 x i32> @llvm.riscv.vle.nxv4i32(<vscale x 4 x i32> undef, ptr %vecaddr1, i64 %vl)
  %sum = add <vscale x 4 x i32> %vec0, %vec1
  ret <vscale x 4 x i32> %sum
}

define void @vle_then_vse(ptr align 4 %srcbase, ptr align 4 %dstbase, i64 %srcoff, i64 %dstoff, i64 %vl) nounwind {
; CHECK-LABEL: vle_then_vse:
; CHECK:       add [[VBASE:[as][0-9]+|t[1-6]|zero|ra|sp|gp|tp|fp]], a0, a2
; CHECK:       vle32.v [[VEC:v[0-9]+]], ([[VBASE]])
; CHECK:       add t0, a1, a3
; CHECK:       vse32.v [[VEC]], (t0)
; CHECK-NEXT:  fence rw, rw
entry:
  %srcaddr = getelementptr i8, ptr %srcbase, i64 %srcoff
  %vec = call <vscale x 4 x i32> @llvm.riscv.vle.nxv4i32(<vscale x 4 x i32> undef, ptr %srcaddr, i64 %vl)
  %dstaddr = getelementptr i8, ptr %dstbase, i64 %dstoff
  call void @llvm.riscv.vse.nxv4i32(<vscale x 4 x i32> %vec, ptr %dstaddr, i64 %vl)
  ret void
}

define i32 @vle_direct_base_then_lw(ptr align 4 %vecbase, ptr align 4 %scalbase, i64 %vl) nounwind {
; CHECK-LABEL: vle_direct_base_then_lw:
; CHECK:       vle32.v v{{[0-9]+}}, (a0)
; Keep the direct/load base in a0 live until vmv.x.s. The younger scalar
; address must therefore remain in a1 instead of reusing a0.
; CHECK-NEXT:  addi a1, a1, 2047
; CHECK-NEXT:  lw a1, 1(a1)
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

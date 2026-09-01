// RUN: buddy-opt %s --convert-linalg-to-gemmini="elem_t=f32 acc_t=f32" | FileCheck %s --check-prefix=CONVERT
// RUN: buddy-opt %s --convert-linalg-to-gemmini="elem_t=f32 acc_t=f32" --convert-linalg-to-loops --lower-gemmini="dim=4 elem_t=f32 acc_t=f32" | FileCheck %s --check-prefix=LOWER

func.func @test_static_matmul_f32(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) {
  // CONVERT-LABEL: func.func @test_static_matmul_f32
  // CONVERT-NOT: bTranspose = true
  // CONVERT: gemmini.tile_matmul %{{.*}} %{{.*}} %{{.*}} %{{.*}} {fullC = true
  // CONVERT-SAME: memref<8x8xf32>
  // CONVERT-SAME: memref<8x8xf32>
  // CONVERT-SAME: memref<8x8xf32>
  // CONVERT-SAME: memref<8x8xf32>
  // LOWER-LABEL: llvm.func @test_static_matmul_f32
  // LOWER: %[[CONFIG_EX_OFF:.+]] = llvm.mlir.constant(4575657221408489476 : i64) : i64
  // LOWER: "gemmini.intr.config_ex"(%[[CONFIG_EX_OFF]], %{{.*}}) : (i64, i64) -> ()
  // LOWER: "gemmini.intr.config_st"
  // LOWER: "gemmini.intr.config_ld"
  // LOWER: "gemmini.intr.loop_ws_config_strides_ab"
  // LOWER: %[[NO_TRANSPOSE_BITS:.+]] = llvm.mlir.constant(0 : i64) : i64
  // LOWER: "gemmini.intr.loop_ws"(%{{.*}}, %[[NO_TRANSPOSE_BITS]]) : (i64, i64) -> ()
  linalg.matmul ins(%A, %B : memref<8x8xf32>, memref<8x8xf32>)
                outs(%C : memref<8x8xf32>)
  return
}

func.func @test_transpose_like_rhs_f32(%A: memref<16x32xf32>, %B: memref<32x64xf32, strided<[1, 768]>>, %C: memref<16x64xf32>) {
  // CONVERT-LABEL: func.func @test_transpose_like_rhs_f32
  // CONVERT: gemmini.tile_matmul %{{.*}} %{{.*}} %{{.*}} %{{.*}} {bTranspose = true, fullC = true}
  // CONVERT-SAME: memref<16x32xf32>
  // CONVERT-SAME: memref<32x64xf32, strided<[1, 768]>>
  // CONVERT-SAME: memref<16x64xf32>
  // CONVERT-SAME: memref<16x64xf32>
  // LOWER-LABEL: llvm.func @test_transpose_like_rhs_f32
  // LOWER: %[[CONFIG_EX_ON:.+]] = llvm.mlir.constant(4575657221408489988 : i64) : i64
  // LOWER: "gemmini.intr.config_ex"(%[[CONFIG_EX_ON]], %{{.*}}) : (i64, i64) -> ()
  // LOWER: "gemmini.intr.config_st"
  // LOWER: "gemmini.intr.config_ld"
  // LOWER: %[[B_ELEM_BYTES:.+]] = llvm.mlir.constant(4 : i64) : i64
  // LOWER: %[[B_STRIDE_BYTES:.+]] = llvm.mul %[[B_INNER_STRIDE:.+]], %[[B_ELEM_BYTES]] : i64
  // LOWER: "gemmini.intr.config_ld"(%{{.*}}, %[[B_STRIDE_BYTES]]) : (i64, i64) -> ()
  // LOWER: "gemmini.intr.loop_ws_config_strides_ab"(%{{.*}}, %[[B_INNER_STRIDE]]) : (i64, i64) -> ()
  // LOWER: %[[TRANSPOSE_BITS:.+]] = llvm.mlir.constant(2 : i64) : i64
  // LOWER: "gemmini.intr.loop_ws"(%{{.*}}, %[[TRANSPOSE_BITS]]) : (i64, i64) -> ()
  linalg.matmul ins(%A, %B : memref<16x32xf32>, memref<32x64xf32, strided<[1, 768]>>)
                outs(%C : memref<16x64xf32>)
  return
}

func.func @test_partial_static_rhs_f32(%A: memref<16x32xf32>, %B: memref<32x64xf32, strided<[1, ?], offset: ?>>, %C: memref<16x64xf32>) {
  // CONVERT-LABEL: func.func @test_partial_static_rhs_f32
  // CONVERT: gemmini.tile_matmul %{{.*}} %{{.*}} %{{.*}} %{{.*}} {bTranspose = true, fullC = true}
  // CONVERT-SAME: memref<16x32xf32>
  // CONVERT-SAME: memref<32x64xf32, strided<[1, ?], offset: ?>>
  // CONVERT-SAME: memref<16x64xf32>
  // CONVERT-SAME: memref<16x64xf32>
  linalg.matmul ins(%A, %B : memref<16x32xf32>, memref<32x64xf32, strided<[1, ?], offset: ?>>)
                outs(%C : memref<16x64xf32>)
  return
}

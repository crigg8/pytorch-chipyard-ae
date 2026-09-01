// RUN: buddy-opt %s --convert-linalg-to-gemmini="elem_t=bf16 acc_t=f32" | FileCheck %s --check-prefix=CONVERT
// RUN: buddy-opt %s --convert-linalg-to-gemmini="elem_t=bf16 acc_t=f32" --convert-linalg-to-loops --lower-gemmini="dim=4 elem_t=bf16 acc_t=f32" | FileCheck %s --check-prefix=LOWER

func.func @test_static_matmul_bf16(%A: memref<8x8xbf16>, %B: memref<8x8xbf16>, %C: memref<8x8xf32>) {
  // CONVERT-LABEL: func.func @test_static_matmul_bf16
  // CONVERT: gemmini.tile_matmul %{{.*}} %{{.*}} %{{.*}} %{{.*}} {fullC = true
  // CONVERT-SAME: memref<8x8xbf16>
  // CONVERT-SAME: memref<8x8xbf16>
  // CONVERT-SAME: memref<8x8xf32>
  // CONVERT-SAME: memref<8x8xf32>
  // LOWER-LABEL: llvm.func @test_static_matmul_bf16
  // LOWER: "gemmini.intr.config_st"
  // LOWER: "gemmini.intr.config_ld"
  // LOWER: "gemmini.intr.loop_ws_config_strides_ab"
  // LOWER: %[[NO_TRANSPOSE_BITS:.+]] = llvm.mlir.constant(0 : i64) : i64
  // LOWER: "gemmini.intr.loop_ws"(%{{.*}}, %[[NO_TRANSPOSE_BITS]]) : (i64, i64) -> ()
  linalg.matmul ins(%A, %B : memref<8x8xbf16>, memref<8x8xbf16>)
                outs(%C : memref<8x8xf32>)
  return
}

func.func @test_batch_matmul_bf16(%A: memref<2x8x8xbf16>, %B: memref<2x8x8xbf16>, %C: memref<2x8x8xf32>) {
  // CONVERT-LABEL: func.func @test_batch_matmul_bf16
  // CONVERT: gemmini.tile_matmul %{{.*}} %{{.*}} %{{.*}} %{{.*}} {fullC = true
  // CONVERT-SAME: memref<8x8xbf16
  // CONVERT-SAME: memref<8x8xbf16
  // CONVERT-SAME: memref<8x8xf32
  // CONVERT-SAME: memref<8x8xf32>
  linalg.batch_matmul ins(%A, %B : memref<2x8x8xbf16>, memref<2x8x8xbf16>)
                      outs(%C : memref<2x8x8xf32>)
  return
}

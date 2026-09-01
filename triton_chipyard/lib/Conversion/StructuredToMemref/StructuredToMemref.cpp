//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR//MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#define DEBUG_TYPE "structured-to-memref"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

static const std::string WRAP_SIDE_BY_SIDE = "wrap_side_by_side";
static const std::string WRAP_STACKED = "wrap_stacked";
static const std::string WRAP_LINEAR = "wrap_linear";

static Value createNeedsPaddingCondition(Location loc, ArrayRef<int64_t> shape,
                                         ArrayRef<OpFoldResult> mixedDims,
                                         ArrayRef<int64_t> staticMaskDims,
                                         OpBuilder &builder);
static Value createFullTileCondition(Location loc, ArrayRef<int64_t> shape,
                                     ArrayRef<OpFoldResult> mixedDims,
                                     ArrayRef<int64_t> staticMaskDims,
                                     OpBuilder &builder);
static FailureOr<bool> getConstantBoolValue(Value value);

static memref::SubViewOp getSubview(int rank, ArrayRef<OpFoldResult> dims,
                                    Value source, Location loc, OpBuilder &b) {
  auto sourceType = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(sourceType, offsets, dims, strides);

  return b.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType), source,
                                     offsets, dims, strides);
}

static Type getElementTypeStructuredPtr(tts::MakeTensorPtrOp op) {
  assert(!op.isBlockPtr());
  // tensor<1024x!tt.ptr<f32>>
  auto ptrType = cast<triton::PointerType>(
      cast<RankedTensorType>(op.getType()).getElementType());
  return ptrType.getPointeeType();
}

static Type getElementTypeBlockPtr(tts::MakeTensorPtrOp op) {
  assert(op.isBlockPtr());
  // !tt.ptr<tensor<128x64xbf16>, 1>
  auto shapedType = cast<ShapedType>(
      cast<triton::PointerType>(op.getType()).getPointeeType());
  return shapedType.getElementType();
}

static MemRefType getResultMemrefType(tts::MakeTensorPtrOp op, int64_t offset,
                                      ArrayRef<int64_t> staticStrides,
                                      ArrayRef<int64_t> resultShape) {
  auto layout = StridedLayoutAttr::get(op.getContext(), offset, staticStrides);
  Type elemType;
  if (op.isBlockPtr()) {
    elemType = getElementTypeBlockPtr(op);
  } else {
    elemType = getElementTypeStructuredPtr(op);
  }
  return MemRefType::get(resultShape, elemType, layout);
}

static MemRefType getResultMemrefType(tts::MakeGatherScatterTensorPtrOp op,
                                      int64_t offset,
                                      ArrayRef<int64_t> staticStrides,
                                      ArrayRef<int64_t> resultShape) {
  auto layout = StridedLayoutAttr::get(op.getContext(), offset, staticStrides);

  auto ptrType = cast<triton::PointerType>(op.getType());
  Type elemType = ptrType.getPointeeType();

  Type realEltTy = cast<RankedTensorType>(elemType).getElementType();
  return MemRefType::get(resultShape, realEltTy, layout);
}

static SmallVector<int64_t>
getWrappedResultShape(ArrayRef<int64_t> resultShape) {
  return SmallVector<int64_t>(resultShape.size(), ShapedType::kDynamic);
}

static SmallVector<int64_t>
getLayoutClassStaticStrides(ArrayRef<int64_t> shape,
                            ArrayRef<OpFoldResult> mixedStrides) {
  SmallVector<int64_t> staticStrides(shape.size(), ShapedType::kDynamic);
  std::optional<unsigned> primaryUnitAxis;

  for (auto [idx, mixedStride] : llvm::enumerate(mixedStrides)) {
    auto stride = getIntAttr(mixedStride);
    if (!stride || stride.value() != 1)
      continue;

    if (shape[idx] == 1) {
      staticStrides[idx] = 1;
      continue;
    }

    if (!primaryUnitAxis) {
      primaryUnitAxis = idx;
      staticStrides[idx] = 1;
    }
  }

  return staticStrides;
}

static SmallVector<int64_t>
getLayoutClassStaticStrides(ArrayRef<int64_t> shape, MemRefType memrefType) {
  auto [sourceStrides, offset] = memrefType.getStridesAndOffset();
  (void)offset;

  SmallVector<int64_t> staticStrides(shape.size(), ShapedType::kDynamic);
  std::optional<unsigned> primaryUnitAxis;

  for (auto [idx, stride] : llvm::enumerate(sourceStrides)) {
    if (stride != 1)
      continue;

    if (shape[idx] == 1) {
      staticStrides[idx] = 1;
      continue;
    }

    if (!primaryUnitAxis) {
      primaryUnitAxis = idx;
      staticStrides[idx] = 1;
    }
  }

  return staticStrides;
}

static MemRefType
getWrappedResultMemrefType(tts::MakeTensorPtrOp op, ArrayRef<int64_t> resultShape,
                           ArrayRef<OpFoldResult> mixedStrides) {
  return getResultMemrefType(
      op, /*offset=*/ShapedType::kDynamic,
      /*staticStrides=*/getLayoutClassStaticStrides(resultShape, mixedStrides),
      /*resultShape=*/resultShape);
}

static SmallVector<int64_t>
getCompactStrideValues(ArrayRef<int64_t> shape,
                       ArrayRef<int64_t> staticStrideLayoutClass) {
  const int64_t rank = shape.size();
  SmallVector<int64_t> strides(rank, 1);
  if (rank == 0)
    return strides;

  std::optional<unsigned> primaryUnitAxis;
  for (auto [idx, stride] : llvm::enumerate(staticStrideLayoutClass)) {
    if (stride == 1 && shape[idx] != 1) {
      primaryUnitAxis = idx;
      break;
    }
  }

  SmallVector<unsigned> order;
  if (primaryUnitAxis)
    order.push_back(*primaryUnitAxis);
  for (int64_t idx = rank - 1; idx >= 0; --idx) {
    if (primaryUnitAxis && idx == *primaryUnitAxis)
      continue;
    order.push_back(static_cast<unsigned>(idx));
  }

  int64_t runningStride = 1;
  for (unsigned axis : order) {
    strides[axis] = runningStride;
    runningStride *= shape[axis];
  }

  for (auto [idx, stride] : llvm::enumerate(staticStrideLayoutClass)) {
    if (stride == 1 && shape[idx] == 1)
      strides[idx] = 1;
  }

  return strides;
}

static int64_t getStaticElementCount(ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t extent : shape) {
    assert(extent != ShapedType::kDynamic &&
           "expected static tensor shape for structured load temporary");
    count *= extent;
  }
  return count;
}

// If there are dimensions with size 1 and stride 0, replace 0 stride with
// the product of sizes of all lower dimensions. This avoids creating memref
// with zero stride.
template <class OpType>
llvm::SmallVector<OpFoldResult> getMixedStridesForMemref(OpType op,
                                                         OpBuilder &b) {
  llvm::SmallVector<OpFoldResult> strides;
  auto accumulate = 1;
  for (auto [size, stride] :
       llvm::reverse(llvm::zip(op.getSizes(), op.getMixedStrides()))) {
    auto strideIntAttr = getIntAttr(stride);
    if (size == 1 && strideIntAttr && strideIntAttr.value() == 0) {
      strides.push_back(b.getIndexAttr(accumulate));
    } else if (auto v = llvm::dyn_cast_if_present<Value>(stride)) {
      OpFoldResult result = getAsOpFoldResult(v);
      strides.push_back(result);
    } else {
      strides.push_back(stride);
    }
    accumulate *= size;
  }
  std::reverse(strides.begin(), strides.end());
  return strides;
}

static OpFoldResult accumulateTargetOffset(Location loc,
                                           ArrayRef<OpFoldResult> offsets,
                                           OpBuilder &b) {
  OpFoldResult targetOffset = b.getIndexAttr(0);
  for (auto o : offsets) {
    targetOffset = addOFRs(targetOffset, o, loc, b);
  }
  return targetOffset;
}

static OpFoldResult accumulateTargetOffset(Location loc,
                                           ArrayRef<OpFoldResult> offsets,
                                           ArrayRef<OpFoldResult> strides,
                                           int gatherDim, OpBuilder &b) {
  OpFoldResult targetOffset = b.getIndexAttr(0);
  for (int i = 0; i < offsets.size(); i++) {

    OpFoldResult offset = offsets[i];
    // If this is the gather dimension, multiply the offset by the stride.
    // Non-gather dimensions are already multiplied by the stride
    // in the offsets in PtrAnalysis.
    if (i == gatherDim) {
      OpFoldResult stride = strides[i];
      offset = mulOFRs(offset, stride, loc, b);
    }
    targetOffset = addOFRs(targetOffset, offset, loc, b);
  }
  return targetOffset;
}

static Value rewriteGatherScatterPtrElement(
    ArrayRef<int64_t> resultShape, tts::MakeGatherScatterTensorPtrOp op,
    Value basePtr, Value gatherOffsetElt, int gatherDim,
    ConversionPatternRewriter &rewriter) {

  auto mixedStrides = getMixedStridesForMemref(op, rewriter);
  SmallVector<int64_t> staticStrides;
  SmallVector<Value> dynamicStrides;
  dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

  auto offsets = op.getMixedOffsets();
  offsets[gatherDim] = gatherOffsetElt;
  auto targetOffset = accumulateTargetOffset(op.getLoc(), offsets, mixedStrides,
                                             gatherDim, rewriter);

  auto staticTargetOffset = getIntAttr(targetOffset);
  auto resultType =
      getResultMemrefType(op, staticTargetOffset.value_or(ShapedType::kDynamic),
                          staticStrides, resultShape);

  std::vector<int64_t> staticSizes = op.getSizes();
  staticSizes[gatherDim] = 1;
  SmallVector<Value> dynSizes; // sizes are always static
  auto sizes = mlir::getMixedValues(staticSizes, dynSizes, rewriter);

  auto castOp = rewriter.create<memref::ReinterpretCastOp>(
      op.getLoc(), resultType, basePtr, targetOffset, sizes, mixedStrides);

  return castOp.getResult();
}

static Value getMaskDimValue(OpFoldResult mixedDim, int64_t staticMaskDim,
                             Location loc, OpBuilder &builder) {
  if (auto value = dyn_cast<Value>(mixedDim))
    return value;
  return builder.create<arith::ConstantIndexOp>(loc, staticMaskDim);
}

static Value createNeedsPaddingCondition(Location loc, ArrayRef<int64_t> shape,
                                         ArrayRef<OpFoldResult> mixedDims,
                                         ArrayRef<int64_t> staticMaskDims,
                                         OpBuilder &builder) {
  Value needsPadding = builder.create<arith::ConstantOp>(
      loc, builder.getBoolAttr(false));
  for (auto [idx, extent] : llvm::enumerate(shape)) {
    Value dim = getMaskDimValue(mixedDims[idx], staticMaskDims[idx], loc,
                                builder);
    Value bound = builder.create<arith::ConstantIndexOp>(loc, extent);
    Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                              dim, bound);
    needsPadding = builder.create<arith::OrIOp>(loc, needsPadding, cmp);
  }
  return needsPadding;
}

static Value createFullTileCondition(Location loc, ArrayRef<int64_t> shape,
                                     ArrayRef<OpFoldResult> mixedDims,
                                     ArrayRef<int64_t> staticMaskDims,
                                     OpBuilder &builder) {
  Value isFullTile =
      builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(true));
  for (auto [idx, extent] : llvm::enumerate(shape)) {
    Value dim = getMaskDimValue(mixedDims[idx], staticMaskDims[idx], loc,
                                builder);
    Value bound = builder.create<arith::ConstantIndexOp>(loc, extent);
    Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                              dim, bound);
    isFullTile = builder.create<arith::AndIOp>(loc, isFullTile, cmp);
  }
  return isFullTile;
}

static FailureOr<bool> getConstantBoolValue(Value value) {
  auto constantOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return failure();
  auto attr = dyn_cast<IntegerAttr>(constantOp.getValue());
  if (!attr || attr.getValue().getBitWidth() != 1)
    return failure();
  return attr.getValue().getBoolValue();
}

// Fill load destination with other value for mask.
static void fillWithValue(Location loc, Value alloc, Value other,
                          ArrayRef<int64_t> shape,
                          SmallVector<OpFoldResult> &&mixedDims,
                          ArrayRef<int64_t> staticMaskDims,
                          ConversionPatternRewriter &rewriter) {
  Value needsPadding = createNeedsPaddingCondition(loc, shape, mixedDims,
                                                   staticMaskDims, rewriter);

  // condition the memset on the padding predicate
  // initialize with padding prior to CopyOp
  rewriter.create<scf::IfOp>(loc, needsPadding, [&](OpBuilder &b, Location loc) {
    b.create<linalg::FillOp>(loc, ValueRange{other}, ValueRange{alloc});
    b.create<scf::YieldOp>(loc);
  });
}

namespace {

struct MakeTensorPtrConverter
    : public OpConversionPattern<tts::MakeTensorPtrOp> {
private:
  using OpConversionPattern<tts::MakeTensorPtrOp>::OpConversionPattern;

  static Type getElementTypeStructuredPtr(tts::MakeTensorPtrOp op) {
    assert(!op.isBlockPtr());
    // tensor<1024x!tt.ptr<f32>>
    auto ptrType = cast<triton::PointerType>(
        cast<RankedTensorType>(op.getType()).getElementType());
    return ptrType.getPointeeType();
  }

  static Type getElementTypeBlockPtr(tts::MakeTensorPtrOp op) {
    assert(op.isBlockPtr());
    // !tt.ptr<tensor<128x64xbf16>, 1>
    auto shapedType = cast<ShapedType>(
        cast<triton::PointerType>(op.getType()).getPointeeType());
    return shapedType.getElementType();
  }

  static MemRefType getResultMemrefType(tts::MakeTensorPtrOp op, int64_t offset,
                                        ArrayRef<int64_t> staticStrides,
                                        ArrayRef<int64_t> resultShape) {
    auto layout =
        StridedLayoutAttr::get(op.getContext(), offset, staticStrides);
    Type elemType;
    if (op.isBlockPtr()) {
      elemType = getElementTypeBlockPtr(op);
    } else {
      elemType = getElementTypeStructuredPtr(op);
    }
    return MemRefType::get(resultShape, elemType, layout);
  }

  std::pair<memref::ReinterpretCastOp, memref::ReinterpretCastOp>
  createSideBySideCastOps(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto resultShape = cast<RankedTensorType>(op.getType()).getShape();

    auto targetOffset = ofrToIndexValue(
        accumulateTargetOffset(op.getLoc(), op.getMixedOffsets(), rewriter),
        loc, rewriter);

    ////////////////////////////////////////////////////////////////////////////
    //
    // Handling side-by-side wraparound
    //
    // Note: We do not support cases where the target has already overflown the
    // number of columns! This is because in PtrAnalysis, the offset has already
    // been collapsed into a single dimension, so it is ambiguous to determine
    // whether the offset actually overflows or just refers to an element on the
    // subsequent rows.
    //
    // Same limitations apply to the stacked wraparound case.
    //
    ////////////////////////////////////////////////////////////////////////////
    //
    //    nextOffset - targetOffset = colSize
    //    d1 + d2 = colSize
    //                          N
    //                                x            clampedOffset
    //      --------------------------*----------------*-----*
    //      |                                          |     nextOffset (might
    //      |                    targetOffset          |             overflow)
    //  y   *-----                    *----------------|
    //      |    |                    |                |
    //  M   |-----                    -----------------|
    //      | d2                              d1       |
    //      --------------------------------------------
    //
    //    x = targetOffset % N
    //    nextOffset = x + colSize
    //    clampedOffset = min(nextOffset, N)
    //    d1 = clampedOffset - x
    //
    ////////////////////////////////////////////////////////////////////////////

    auto resultType = getResultMemrefType(
        op, /* offset */ ShapedType::kDynamic,
        /* staticStrides */
        getLayoutClassStaticStrides(resultShape, getMixedStridesForMemref(op, rewriter)),
        /* result shape */ getWrappedResultShape(resultShape));

    Value rowSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[0]));
    Value colSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[1]));

    auto mixedStrides = getMixedStridesForMemref(op, rewriter);
    SmallVector<Value> strideVals = ofrsToIndexValues(mixedStrides, loc, rewriter);
    Value strideCol = strideVals[1];

    // Split pointers keep the wrap bound in linearized element units.
    Value modNLinear = ofrToIndexValue(op.getMixedShape()[1], loc, rewriter);
    Value modN = rewriter.create<arith::DivSIOp>(loc, modNLinear, strideCol);

    Value xLinear = rewriter.create<arith::RemSIOp>(loc, targetOffset, modNLinear);
    Value y = rewriter.create<arith::SubIOp>(loc, targetOffset, xLinear);
    Value x = rewriter.create<arith::DivSIOp>(loc, xLinear, strideCol);

    // First chunk
    Value nextOffset = rewriter.create<arith::AddIOp>(loc, x, colSize);
    Value clampedOffset =
        rewriter.create<arith::MinSIOp>(loc, nextOffset, modN);
    Value d1 = rewriter.create<arith::SubIOp>(loc, clampedOffset, x);
    SmallVector<OpFoldResult> sizes1{rowSize, d1};

    auto cast1 = rewriter.create<memref::ReinterpretCastOp>(
        loc, resultType, adaptor.getBase(), getAsOpFoldResult(targetOffset), sizes1,
        mixedStrides);

    // Second chunk
    Value d2 = rewriter.create<arith::SubIOp>(loc, colSize, d1);
    SmallVector<OpFoldResult> sizes2{rowSize, d2};

    auto cast2 = rewriter.create<memref::ReinterpretCastOp>(
        loc, resultType, adaptor.getBase(), getAsOpFoldResult(y), sizes2,
        mixedStrides);

    return {cast1, cast2};
  }

  SmallVector<Value>
  createLinearWrapOperands(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto resultShape = cast<RankedTensorType>(op.getType()).getShape();

    assert(resultShape.size() == 1);

    Value stride = ofrToIndexValue(op.getMixedStrides()[0], loc, rewriter);
    Value wrapBound = ofrToIndexValue(op.getMixedShape()[0], loc, rewriter);

    Value targetOffset = ofrToIndexValue(
        accumulateTargetOffset(op.getLoc(), op.getMixedOffsets(), rewriter), loc,
        rewriter);
    Value wrappedOffset =
        rewriter.create<arith::RemSIOp>(loc, targetOffset, wrapBound);

    // A linear wraparound is a circular access pattern rather than a single
    // strided memref view. Keep only the metadata needed to materialize it
    // during load lowering instead of exploding it into many chunk casts.
    return {adaptor.getBase(), wrappedOffset, wrapBound, stride};
  }

  std::pair<memref::ReinterpretCastOp, memref::ReinterpretCastOp>
  createStackedCastOps(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {

    auto loc = op->getLoc();
    auto resultShape = cast<RankedTensorType>(op.getType()).getShape();

    assert(resultShape.size() == 2);

    auto targetOffset = ofrToIndexValue(
        accumulateTargetOffset(op.getLoc(), op.getMixedOffsets(), rewriter),
        loc, rewriter);

    ////////////////////////////////////////////////////////////////////////////
    //
    // Handling stacked wraparound
    //
    // We do not support cases where the target offset has already overflown the
    // number of rows. See side-by-side wraparound for details.
    //
    ////////////////////////////////////////////////////////////////////////////
    //    We're loading a tensor of dim (rowSize, colSize)
    //    d1 + d2 = rowSize
    //    d2 is the number of rows that overflow
    //
    //                       cols
    //
    //               wrappedAroundOff
    //      --------------*------------*--------
    //      |        d2   |            |       |
    //      |             |------------|       |
    //  rows|                                  |
    //      |                                  |
    //      |           targetOffset           |
    //      |             *------------|       |
    //      |             |            |       |
    //      |         d1  |            |       |
    //      |             | clampedOff |       |
    //      --------------*---------------------
    //                    |  overflow  |
    //                    *-------------
    //                 nextOff
    //
    //    wrappedAroundOff = targetOffset % cols
    //    clampedOff = (rows * strideRows) + wrappedAroundOff
    //                  ~~~~~~~~~~~~~~~~~
    //                         ^
    //                         |
    //          We have already computed
    //          rows * strideRows = modRow = shape[1]
    //          in TritonToStructured
    //
    //          clampedOff - targetOffset
    //    d1 = --------------------
    //              strideRows
    //
    ////////////////////////////////////////////////////////////////////////////
    //
    //                       cols
    //
    //               wrappedAroundOff
    //      --------------*---------------------
    //      |                                  |
    //      |           targetOffset           |
    //      |             *------------|       |
    //      |             |            |       |
    //      |             |            |       |
    //  rows|    rowSize  |            |       |
    //      |             |            |       |
    //      |             |            |       |
    //      |             *------------|       |
    //      |          nextOff                 |
    //      |                                  |
    //      |          clampedOff              |
    //      --------------*---------------------
    //
    //    For the case that clampedOff is not overflown
    //    d1 = min(d1, rowSize)
    //

    auto resultType = getResultMemrefType(
        op, /* offset */ ShapedType::kDynamic,
        /* staticStrides */
        getLayoutClassStaticStrides(resultShape, getMixedStridesForMemref(op, rewriter)),
        /* result shape */ getWrappedResultShape(resultShape));

    Value rowSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[0]));
    Value colSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[1]));

    auto mixedStrides = getMixedStridesForMemref(op, rewriter);
    Value strideRow = ofrToIndexValue(mixedStrides[0], loc, rewriter);
    Value strideCol = ofrToIndexValue(mixedStrides[1], loc, rewriter);

    Value modRow = ofrToIndexValue(op.getMixedShape()[0], loc, rewriter);

    // First chunk
    Value wrappedAroundOff =
        rewriter.create<arith::RemSIOp>(loc, targetOffset, strideRow);
    Value clampedOff =
        rewriter.create<arith::AddIOp>(loc, modRow, wrappedAroundOff);
    Value d1 = rewriter.create<arith::SubIOp>(loc, clampedOff, targetOffset);
    d1 = rewriter.create<arith::DivSIOp>(loc, d1, strideRow);
    d1 = rewriter.create<arith::MinSIOp>(loc, d1, rowSize);

    SmallVector<OpFoldResult> sizes1{d1, colSize};
    memref::ReinterpretCastOp cast1 =
        rewriter.create<memref::ReinterpretCastOp>(
            loc, resultType, adaptor.getBase(), getAsOpFoldResult(targetOffset),
            sizes1, mixedStrides);

    // Second chunk
    Value d2 = rewriter.create<arith::SubIOp>(loc, rowSize, d1);
    SmallVector<OpFoldResult> sizes2{d2, colSize};
    memref::ReinterpretCastOp cast2 =
        rewriter.create<memref::ReinterpretCastOp>(
            loc, resultType, adaptor.getBase(),
            getAsOpFoldResult(wrappedAroundOff), sizes2, mixedStrides);

    return {cast1, cast2};
  }

  LogicalResult rewriteSplitPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto parentShape = op.getStaticShape();
    if (parentShape.size() != 1 && parentShape.size() != 2) {
      return rewriter.notifyMatchFailure(
          op, "split pointer is only supported for rank-1 or rank-2 tensors");
    }
    SmallVector<Value> casts;
    StringRef wrapType;
    auto mixedStrides = getMixedStridesForMemref(op, rewriter);
    auto resultShape = cast<RankedTensorType>(op.getType()).getShape();

    if (parentShape.size() == 1) {
      casts = createLinearWrapOperands(op, adaptor, rewriter);
      wrapType = WRAP_LINEAR;
    } else {

      // For split pointers, a split dimension is either a dynamic or a non-zero
      // value. The other dimension must be zero.
      auto isSplitDimension = [](int64_t dim) {
        return dim == ShapedType::kDynamic || dim != 0;
      };

      if (isSplitDimension(parentShape[0])) {
        // Stacked case
        if (parentShape[1] != 0) {
          return rewriter.notifyMatchFailure(
              op, "invalid split-pointer shape for stacked wrapping");
        }
        auto [cast1, cast2] = createStackedCastOps(op, adaptor, rewriter);
        casts = {cast1.getResult(), cast2.getResult()};
        wrapType = WRAP_STACKED;
      } else if (isSplitDimension(parentShape[1])) {
        if (parentShape[0] != 0) {
          return rewriter.notifyMatchFailure(
              op, "invalid split-pointer shape for side-by-side wrapping");
        }
        auto [cast1, cast2] = createSideBySideCastOps(op, adaptor, rewriter);
        casts = {cast1.getResult(), cast2.getResult()};
        wrapType = WRAP_SIDE_BY_SIDE;
      } else {
        return rewriter.notifyMatchFailure(op, "unexpected split-pointer shape");
      }
    }

    Type splitResultType = op.getType();
    if (wrapType == WRAP_LINEAR) {
      const TypeConverter *converter = getTypeConverter();
      if (!converter) {
        return rewriter.notifyMatchFailure(
            op, "missing type converter for wrapped split pointer");
      }
      Type converted = converter->convertType(op.getType());
      if (!converted || !isa<MemRefType>(converted)) {
        return rewriter.notifyMatchFailure(
            op, "failed to convert wrapped split-pointer result to memref");
      }
      splitResultType = converted;
    } else if (wrapType == WRAP_SIDE_BY_SIDE || wrapType == WRAP_STACKED) {
      if (casts.empty() || !llvm::all_of(casts, [](Value cast) {
            return isa<MemRefType>(cast.getType());
          })) {
        return rewriter.notifyMatchFailure(
            op, "expected memref chunk casts for wrapped split pointer");
      }
      splitResultType =
          getWrappedResultMemrefType(op, resultShape, mixedStrides);
    }

    auto combinedCast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), splitResultType, casts);

    combinedCast->setAttr(wrapType, rewriter.getUnitAttr());

    rewriter.replaceOp(op, combinedCast.getResults());

    return success();
  }

  LogicalResult rewritePtr(ArrayRef<int64_t> resultShape, bool isBlockPtr,
                           tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {

    auto mixedStrides = getMixedStridesForMemref(op, rewriter);
    SmallVector<int64_t> staticStrides;
    SmallVector<Value> dynamicStrides;
    dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

    auto targetOffset =
        accumulateTargetOffset(op.getLoc(), op.getMixedOffsets(), rewriter);
    auto staticTargetOffset = getIntAttr(targetOffset);
    auto resultType = getResultMemrefType(
        op, staticTargetOffset.value_or(ShapedType::kDynamic), staticStrides,
        resultShape);

    auto castOp = rewriter.create<memref::ReinterpretCastOp>(
        op.getLoc(), resultType, adaptor.getBase(), targetOffset,
        op.getMixedSizes(), mixedStrides);

    rewriter.replaceOp(op, castOp);

    return success();
  }

  LogicalResult
  rewriteStructuredPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    ArrayRef<int64_t> resultShape = cast<ShapedType>(op.getType()).getShape();
    return rewritePtr(resultShape, false, op, adaptor, rewriter);
  }

  LogicalResult rewriteBlockPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // Block pointers are basically the same as structured pointers except that
    // the return types are !tt.ptr<tensor<AxBxCxbf16>> instead of
    // tensor<AxBxCx!tt.ptr<bf16>>
    ArrayRef<int64_t> resultShape =
        cast<ShapedType>(
            cast<triton::PointerType>(op.getType()).getPointeeType())
            .getShape();
    return rewritePtr(resultShape, true, op, adaptor, rewriter);
  }

public:
  MakeTensorPtrConverter(const TypeConverter &typeConverter,
                         MLIRContext *context)
      : OpConversionPattern<tts::MakeTensorPtrOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!llvm::is_sorted(op.getOrder(), std::greater<>())) {
      emitError(op.getLoc()) << "non-decreasing dimension order on tensor "
                                "pointers are not yet supported";
      return failure();
    }

    if (op.isBlockPtr()) {
      return rewriteBlockPtr(op, adaptor, rewriter);
    }

    if (op.isStructuredPtr()) {
      return rewriteStructuredPtr(op, adaptor, rewriter);
    }

    if (op.isSplitPtr()) {
      return rewriteSplitPtr(op, adaptor, rewriter);
    }

    return failure();
  }
};

struct MakeGatherScatterTensorPtrConverter
    : public OpConversionPattern<tts::MakeGatherScatterTensorPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tts::MakeGatherScatterTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The gatherScatterPtr is rewritten as separate rows during load/store
    // operations. Therefore, no action is needed here except saving
    // adaptor.getBase(). DialectConversion will ignore pure type conversion if
    // we were to simply replace the op with adaptor.getBase(). To circumvent
    // this we create an identity cast.
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, adaptor.getBase().getType(), adaptor.getBase());
    return success();
  }
};

static tts::MakeGatherScatterTensorPtrOp
getGatherScatterPtrDef(Value ptr) {
  if (auto gatherScatterPtr =
          ptr.getDefiningOp<tts::MakeGatherScatterTensorPtrOp>()) {
    return gatherScatterPtr;
  }

  if (auto castOp = ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
    for (Value operand : castOp.getOperands()) {
      if (auto gatherScatterPtr =
              operand.getDefiningOp<tts::MakeGatherScatterTensorPtrOp>()) {
        return gatherScatterPtr;
      }
    }
  }

  return {};
}

static bool areCompatibleMemRefTypes(MemRefType lhs, MemRefType rhs) {
  if (lhs.getRank() != rhs.getRank())
    return false;
  if (lhs.getElementType() != rhs.getElementType())
    return false;
  if (lhs.getMemorySpace() != rhs.getMemorySpace())
    return false;

  for (auto [lhsDim, rhsDim] : llvm::zip(lhs.getShape(), rhs.getShape())) {
    if (!ShapedType::isDynamic(lhsDim) && !ShapedType::isDynamic(rhsDim) &&
        lhsDim != rhsDim)
      return false;
  }

  return true;
}

static Value unwrapMemrefIdentityCasts(Value ptr) {
  Value current = ptr;
  while (auto castOp = current.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (castOp->hasAttr(WRAP_SIDE_BY_SIDE) ||
        castOp->hasAttr(WRAP_STACKED) || castOp->hasAttr(WRAP_LINEAR)) {
      break;
    }
    if (castOp.getNumOperands() != 1) {
      break;
    }
    Value operand = castOp.getOperand(0);
    auto currentType = dyn_cast<MemRefType>(current.getType());
    auto operandType = dyn_cast<MemRefType>(operand.getType());
    if (!currentType || !operandType ||
        !areCompatibleMemRefTypes(currentType, operandType)) {
      break;
    }
    current = operand;
  }
  return current;
}

static UnrealizedConversionCastOp getWrapCast(Value ptr) {
  Value current = ptr;
  while (auto castOp = current.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (castOp->hasAttr(WRAP_SIDE_BY_SIDE) ||
        castOp->hasAttr(WRAP_STACKED) || castOp->hasAttr(WRAP_LINEAR))
      return castOp;
    if (castOp.getNumOperands() != 1)
      break;
    Value operand = castOp.getOperand(0);
    auto currentType = dyn_cast<MemRefType>(current.getType());
    auto operandType = dyn_cast<MemRefType>(operand.getType());
    if (!currentType || !operandType ||
        !areCompatibleMemRefTypes(currentType, operandType)) {
      break;
    }
    current = operand;
  }
  return {};
}

struct LoadConverter : public OpConversionPattern<tts::LoadOp> {
private:
  bool enableAliasFirst;

  bool hasAutomaticAllocationScopeAncestor(tts::LoadOp op) const {
    for (Operation *parent = op->getParentOp(); parent != nullptr;
         parent = parent->getParentOp()) {
      if (parent->hasTrait<OpTrait::AutomaticAllocationScope>())
        return true;
    }
    return false;
  }

  Value createLoadTemporary(tts::LoadOp op, Value layoutSource,
                            RankedTensorType tensorType,
                            Location loc,
                            ConversionPatternRewriter &rewriter) const {
    layoutSource = unwrapMemrefIdentityCasts(layoutSource);
    auto sourceType = dyn_cast<MemRefType>(layoutSource.getType());
    if (!sourceType) {
      auto memrefType =
          MemRefType::get(tensorType.getShape(), tensorType.getElementType());
      if (hasAutomaticAllocationScopeAncestor(op))
        return rewriter.create<memref::AllocaOp>(loc, memrefType);
      return rewriter.create<memref::AllocOp>(loc, memrefType);
    }

    ArrayRef<int64_t> shape = tensorType.getShape();
    SmallVector<int64_t> staticStrides =
        getLayoutClassStaticStrides(shape, sourceType);
    SmallVector<int64_t> compactStrides =
        getCompactStrideValues(shape, staticStrides);
    int64_t elementCount = getStaticElementCount(shape);

    auto backingType =
        MemRefType::get({elementCount}, tensorType.getElementType(),
                        AffineMap(), sourceType.getMemorySpace());
    Value backing;
    if (hasAutomaticAllocationScopeAncestor(op))
      backing = rewriter.create<memref::AllocaOp>(loc, backingType);
    else
      backing = rewriter.create<memref::AllocOp>(loc, backingType);

    auto layoutAwareType = MemRefType::get(
        shape, tensorType.getElementType(),
        StridedLayoutAttr::get(rewriter.getContext(), /*offset=*/0,
                               staticStrides),
        sourceType.getMemorySpace());

    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
    sizes.reserve(shape.size());
    strides.reserve(shape.size());
    for (int64_t extent : shape)
      sizes.push_back(rewriter.getIndexAttr(extent));
    for (auto [idx, stride] : llvm::enumerate(compactStrides)) {
      if (staticStrides[idx] == 1) {
        strides.push_back(rewriter.getIndexAttr(1));
        continue;
      }
      strides.push_back(rewriter.create<arith::ConstantIndexOp>(loc, stride)
                            .getResult());
    }

    return rewriter
        .create<memref::ReinterpretCastOp>(loc, layoutAwareType, backing,
                                           rewriter.getIndexAttr(0),
                                           sizes, strides)
        .getResult();
  }

  MemRefType getLoadResultMemrefType(Value ptr, RankedTensorType tensorType,
                                     ConversionPatternRewriter &rewriter) const {
    Value layoutSource = unwrapMemrefIdentityCasts(ptr);
    auto sourceType = dyn_cast<MemRefType>(layoutSource.getType());
    if (!sourceType)
      return MemRefType::get(tensorType.getShape(), tensorType.getElementType());

    auto layout = StridedLayoutAttr::get(
        rewriter.getContext(), /*offset=*/ShapedType::kDynamic,
        getLayoutClassStaticStrides(tensorType.getShape(), sourceType));
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                           layout, sourceType.getMemorySpace());
  }

  Value castMemrefToTensorShape(Value memref, RankedTensorType tensorType,
                                Location loc,
                                ConversionPatternRewriter &rewriter) const {
    memref = unwrapMemrefIdentityCasts(memref);
    auto memrefType = cast<MemRefType>(memref.getType());
    auto targetType =
        MemRefType::get(tensorType.getShape(), memrefType.getElementType(),
                        memrefType.getLayout(), memrefType.getMemorySpace());
    if (memrefType == targetType)
      return memref;
    return rewriter.create<memref::CastOp>(loc, targetType, memref);
  }

  Value castMemrefToType(Value memref, MemRefType targetType, Location loc,
                         ConversionPatternRewriter &rewriter) const {
    memref = unwrapMemrefIdentityCasts(memref);
    auto memrefType = cast<MemRefType>(memref.getType());
    if (memrefType == targetType)
      return memref;
    return rewriter.create<memref::CastOp>(loc, targetType, memref);
  }

  Value createLoadTensor(Value memref, RankedTensorType tensorType,
                         Location loc,
                         ConversionPatternRewriter &rewriter,
                         bool writable) const {
    return rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, memref, true /* restrict */, writable);
  }

  Value createAliasedLoadTensor(Value memref, RankedTensorType tensorType,
                                Location loc,
                                ConversionPatternRewriter &rewriter) const {
    Value shapedMemref =
        castMemrefToTensorShape(memref, tensorType, loc, rewriter);
    return createLoadTensor(shapedMemref, tensorType, loc, rewriter,
                            false /* writable */);
  }

  Value createAliasedLoadMemref(Value memref, RankedTensorType tensorType,
                                MemRefType resultType, Location loc,
                                ConversionPatternRewriter &rewriter) const {
    Value shapedMemref =
        castMemrefToTensorShape(memref, tensorType, loc, rewriter);
    return castMemrefToType(shapedMemref, resultType, loc, rewriter);
  }

  void createLinearWrapCopy(UnrealizedConversionCastOp wrapCast, Value dst,
                            Value copyLimit, Location loc,
                            ConversionPatternRewriter &rewriter) const {
    auto operands = wrapCast.getOperands();
    assert(operands.size() == 4 &&
           "expected linear wrap metadata operands: base, offset, bound, stride");

    Value basePtr = operands[0];
    Value wrappedOffset = operands[1];
    Value wrapBound = operands[2];
    Value stride = operands[3];

    auto baseType = cast<BaseMemRefType>(basePtr.getType());
    auto linearType = MemRefType::get(
        {ShapedType::kDynamic}, baseType.getElementType(),
        StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic,
                               {ShapedType::kDynamic}),
        baseType.getMemorySpace());

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value linearBase = rewriter
                           .create<memref::ReinterpretCastOp>(
                               loc, linearType, basePtr, zero,
                               ValueRange{wrapBound}, ValueRange{one})
                           .getResult();

    OpBuilder::InsertionGuard guard(rewriter);
    auto loop = rewriter.create<scf::ForOp>(loc, zero, copyLimit, one);
    rewriter.setInsertionPointToStart(loop.getBody());

    Value iv = loop.getInductionVar();
    Value scaledOffset = rewriter.create<arith::MulIOp>(loc, iv, stride);
    Value sourceOffset =
        rewriter.create<arith::AddIOp>(loc, wrappedOffset, scaledOffset);
    Value sourceIndex =
        rewriter.create<arith::RemSIOp>(loc, sourceOffset, wrapBound);
    Value value =
        rewriter.create<memref::LoadOp>(loc, linearBase, ValueRange{sourceIndex});
    rewriter.create<memref::StoreOp>(loc, value, dst, ValueRange{iv});
  }

  void createSideBySideCopies(Value block1, Value block2, Value dst,
                              Location loc,
                              ConversionPatternRewriter &rewriter) const {

    auto zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);

    Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

    auto block1Dst =
        rewriter.create<memref::SubViewOp>(loc, dst, /* offsets */
                                           ValueRange{zero, zero},
                                           /* sizes */
                                           ValueRange{block1Row, block1Col},
                                           /* strides */
                                           ValueRange{one, one});

    auto block2Dst =
        rewriter.create<memref::SubViewOp>(loc, dst,
                                           /* offsets */
                                           ValueRange{zero, block1Col},
                                           /* sizes */
                                           ValueRange{block2Row, block2Col},
                                           /* strides */
                                           ValueRange{one, one});

    rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
    rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
  }

  void createStackedCopies(Value block1, Value block2, Value dst, Location loc,
                           ConversionPatternRewriter &rewriter) const {

    auto zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);

    Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

    auto block1Dst =
        rewriter.create<memref::SubViewOp>(loc, dst, /* offsets */
                                           ValueRange{zero, zero},
                                           /* sizes */
                                           ValueRange{block1Row, block1Col},
                                           /* strides */
                                           ValueRange{one, one});

    auto block2Dst =
        rewriter.create<memref::SubViewOp>(loc, dst,
                                           /* offsets */
                                           ValueRange{block1Row, zero},
                                           /* sizes */
                                           ValueRange{block2Row, block2Col},
                                           /* strides */
                                           ValueRange{one, one});

    rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
    rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
  }

  memref::SubViewOp createSubview(Value src, ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes,
                                  ArrayRef<OpFoldResult> strides, Location loc,
                                  ConversionPatternRewriter &rewriter) const {
    auto srcType = cast<MemRefType>(src.getType());
    auto dstType =
        memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
    return rewriter.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType),
                                              src, offsets, sizes, strides);
  }

  std::pair<memref::SubViewOp, memref::SubViewOp>
  getSideBySideSubviews(ArrayRef<OpFoldResult> dims, Value block1, Value block2,
                        Location loc,
                        ConversionPatternRewriter &rewriter) const {
    OpFoldResult subviewRowFull = dims[0];
    OpFoldResult subviewColFull = dims[1];
    OpFoldResult subviewCol1 =
        rewriter.create<memref::DimOp>(loc, block1, 1).getResult();
    OpFoldResult subviewCol2 =
        rewriter.create<memref::DimOp>(loc, block2, 1).getResult();

    SmallVector<OpFoldResult> offsets(dims.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(dims.size(), rewriter.getIndexAttr(1));
    auto sv1 = createSubview(block1, offsets, {subviewRowFull, subviewCol1},
                             strides, loc, rewriter);
    auto sv2 = createSubview(block2, offsets, {subviewRowFull, subviewCol2},
                             strides, loc, rewriter);

    return {sv1, sv2};
  }

  std::pair<memref::SubViewOp, memref::SubViewOp>
  getStackedSubviews(ArrayRef<OpFoldResult> dims, Value block1, Value block2,
                     const Location loc,
                     ConversionPatternRewriter &rewriter) const {
    OpFoldResult subviewRowFull = dims[0];
    OpFoldResult subviewColFull = dims[1];
    OpFoldResult subviewRow1 =
        rewriter.create<memref::DimOp>(loc, block1, 0).getResult();
    OpFoldResult subviewRow2 =
        rewriter.create<memref::DimOp>(loc, block2, 0).getResult();

    SmallVector<OpFoldResult> offsets(dims.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(dims.size(), rewriter.getIndexAttr(1));
    auto sv1 = createSubview(block1, offsets, {subviewRow1, subviewColFull},
                             strides, loc, rewriter);
    auto sv2 = createSubview(block2, offsets, {subviewRow2, subviewColFull},
                             strides, loc, rewriter);
    return {sv1, sv2};
  }

  Value materializeStructuredLoad(tts::LoadOp op, Value ptr,
                                  RankedTensorType tensorType, Location loc,
                                  ConversionPatternRewriter &rewriter) const {
    Value alloc = createLoadTemporary(op, ptr, tensorType, loc, rewriter);
    if (auto wrapCast = getWrapCast(ptr)) {
      auto memrefs = wrapCast.getOperands();
      if (wrapCast->hasAttr(WRAP_SIDE_BY_SIDE)) {
        assert(memrefs.size() == 2);
        createSideBySideCopies(memrefs[0], memrefs[1], alloc, loc, rewriter);
      } else if (wrapCast->hasAttr(WRAP_STACKED)) {
        assert(memrefs.size() == 2);
        createStackedCopies(memrefs[0], memrefs[1], alloc, loc, rewriter);
      } else if (wrapCast->hasAttr(WRAP_LINEAR)) {
        Value copyLimit = rewriter.create<memref::DimOp>(loc, alloc, 0);
        createLinearWrapCopy(wrapCast, alloc, copyLimit, loc, rewriter);
      } else {
        llvm_unreachable("unexpected wraparound type");
      }
    } else {
      Value source = unwrapMemrefIdentityCasts(ptr);
      rewriter.create<memref::CopyOp>(loc, source, alloc);
    }

    return createLoadTensor(alloc, tensorType, loc, rewriter,
                            true /* writable */);
  }

  Value materializeStructuredLoadMemref(tts::LoadOp op, Value ptr,
                                        RankedTensorType tensorType,
                                        MemRefType resultType, Location loc,
                                        ConversionPatternRewriter &rewriter)
      const {
    Value alloc = createLoadTemporary(op, ptr, tensorType, loc, rewriter);
    if (auto wrapCast = getWrapCast(ptr)) {
      auto memrefs = wrapCast.getOperands();
      if (wrapCast->hasAttr(WRAP_SIDE_BY_SIDE)) {
        assert(memrefs.size() == 2);
        createSideBySideCopies(memrefs[0], memrefs[1], alloc, loc, rewriter);
      } else if (wrapCast->hasAttr(WRAP_STACKED)) {
        assert(memrefs.size() == 2);
        createStackedCopies(memrefs[0], memrefs[1], alloc, loc, rewriter);
      } else if (wrapCast->hasAttr(WRAP_LINEAR)) {
        Value copyLimit = rewriter.create<memref::DimOp>(loc, alloc, 0);
        createLinearWrapCopy(wrapCast, alloc, copyLimit, loc, rewriter);
      } else {
        llvm_unreachable("unexpected wraparound type");
      }
    } else {
      Value source = unwrapMemrefIdentityCasts(ptr);
      rewriter.create<memref::CopyOp>(loc, source, alloc);
    }

    return castMemrefToType(alloc, resultType, loc, rewriter);
  }

  Value materializeMaskedStructuredLoad(tts::LoadOp op, Value ptr,
                                        RankedTensorType tensorType,
                                        ArrayRef<OpFoldResult> mixedDims,
                                        Location loc,
                                        ConversionPatternRewriter &rewriter)
      const {
    Value alloc = createLoadTemporary(op, ptr, tensorType, loc, rewriter);

    if (Value other = op.getOther()) {
      fillWithValue(loc, alloc, other, tensorType.getShape(),
                    SmallVector<OpFoldResult>(mixedDims.begin(),
                                              mixedDims.end()),
                    op.getStaticMaskDims(), rewriter);
    }

    if (auto wrapCast = getWrapCast(ptr)) {
      auto memrefs = wrapCast.getOperands();
      if (wrapCast->hasAttr(WRAP_SIDE_BY_SIDE)) {
        assert(memrefs.size() == 2);
        auto [subview1, subview2] =
            getSideBySideSubviews(mixedDims, memrefs[0], memrefs[1], loc,
                                  rewriter);
        createSideBySideCopies(subview1, subview2, alloc, loc, rewriter);
      } else if (wrapCast->hasAttr(WRAP_STACKED)) {
        assert(memrefs.size() == 2);
        auto [subview1, subview2] =
            getStackedSubviews(mixedDims, memrefs[0], memrefs[1], loc,
                               rewriter);
        createStackedCopies(subview1, subview2, alloc, loc, rewriter);
      } else if (wrapCast->hasAttr(WRAP_LINEAR)) {
        Value copyLimit = ofrToIndexValue(mixedDims[0], loc, rewriter);
        createLinearWrapCopy(wrapCast, alloc, copyLimit, loc, rewriter);
      } else {
        llvm_unreachable("unexpected wraparound type");
      }
    } else {
      Value source = unwrapMemrefIdentityCasts(ptr);
      if (!isa<MemRefType>(source.getType())) {
        llvm_unreachable("expected memref pointer in masked load lowering");
      }
      memref::SubViewOp srcSubview =
          getSubview(tensorType.getRank(), mixedDims, source, loc, rewriter);
      memref::SubViewOp dstSubview =
          getSubview(tensorType.getRank(), mixedDims, alloc, loc, rewriter);
      rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
    }

    return createLoadTensor(alloc, tensorType, loc, rewriter,
                            true /* writable */);
  }

  Value materializeMaskedStructuredLoadMemref(
      tts::LoadOp op, Value ptr, RankedTensorType tensorType,
      ArrayRef<OpFoldResult> mixedDims, MemRefType resultType, Location loc,
      ConversionPatternRewriter &rewriter) const {
    Value alloc = createLoadTemporary(op, ptr, tensorType, loc, rewriter);

    if (Value other = op.getOther()) {
      fillWithValue(loc, alloc, other, tensorType.getShape(),
                    SmallVector<OpFoldResult>(mixedDims.begin(),
                                              mixedDims.end()),
                    op.getStaticMaskDims(), rewriter);
    }

    if (auto wrapCast = getWrapCast(ptr)) {
      auto memrefs = wrapCast.getOperands();
      if (wrapCast->hasAttr(WRAP_SIDE_BY_SIDE)) {
        assert(memrefs.size() == 2);
        auto [subview1, subview2] =
            getSideBySideSubviews(mixedDims, memrefs[0], memrefs[1], loc,
                                  rewriter);
        createSideBySideCopies(subview1, subview2, alloc, loc, rewriter);
      } else if (wrapCast->hasAttr(WRAP_STACKED)) {
        assert(memrefs.size() == 2);
        auto [subview1, subview2] =
            getStackedSubviews(mixedDims, memrefs[0], memrefs[1], loc,
                               rewriter);
        createStackedCopies(subview1, subview2, alloc, loc, rewriter);
      } else if (wrapCast->hasAttr(WRAP_LINEAR)) {
        Value copyLimit = ofrToIndexValue(mixedDims[0], loc, rewriter);
        createLinearWrapCopy(wrapCast, alloc, copyLimit, loc, rewriter);
      } else {
        llvm_unreachable("unexpected wraparound type");
      }
    } else {
      Value source = unwrapMemrefIdentityCasts(ptr);
      if (!isa<MemRefType>(source.getType())) {
        llvm_unreachable("expected memref pointer in masked load lowering");
      }
      memref::SubViewOp srcSubview =
          getSubview(tensorType.getRank(), mixedDims, source, loc, rewriter);
      memref::SubViewOp dstSubview =
          getSubview(tensorType.getRank(), mixedDims, alloc, loc, rewriter);
      rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
    }

    return castMemrefToType(alloc, resultType, loc, rewriter);
  }

  Value lowerStructuredLoadWithAlias(tts::LoadOp op, Value ptr,
                                     RankedTensorType tensorType, Location loc,
                                     ConversionPatternRewriter &rewriter) const {
    auto wrapCast = getWrapCast(ptr);
    if (!wrapCast)
      return createAliasedLoadTensor(ptr, tensorType, loc, rewriter);

    if (wrapCast->hasAttr(WRAP_LINEAR))
      return materializeStructuredLoad(op, ptr, tensorType, loc, rewriter);

    auto memrefs = wrapCast.getOperands();
    assert(memrefs.size() == 2 && "expected two memrefs for split wraparound");
    Value block1 = memrefs[0];
    Value block2 = memrefs[1];

    unsigned splitDim = wrapCast->hasAttr(WRAP_SIDE_BY_SIDE) ? 1 : 0;
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value block1Extent = rewriter.create<memref::DimOp>(loc, block1, splitDim);
    Value block2Extent = rewriter.create<memref::DimOp>(loc, block2, splitDim);
    Value block1Empty = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, block1Extent, zero);
    Value block2Empty = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, block2Extent, zero);

    OpBuilder::InsertionGuard guard(rewriter);
    auto outerIf =
        rewriter.create<scf::IfOp>(loc, TypeRange{tensorType}, block2Empty,
                                   /*withElseRegion=*/true);

    rewriter.setInsertionPointToStart(&outerIf.getThenRegion().front());
    Value block1Tensor = createAliasedLoadTensor(block1, tensorType, loc,
                                                 rewriter);
    rewriter.create<scf::YieldOp>(loc, block1Tensor);

    rewriter.setInsertionPointToStart(&outerIf.getElseRegion().front());
    auto innerIf =
        rewriter.create<scf::IfOp>(loc, TypeRange{tensorType}, block1Empty,
                                   /*withElseRegion=*/true);

    rewriter.setInsertionPointToStart(&innerIf.getThenRegion().front());
    Value block2Tensor = createAliasedLoadTensor(block2, tensorType, loc,
                                                 rewriter);
    rewriter.create<scf::YieldOp>(loc, block2Tensor);

    rewriter.setInsertionPointToStart(&innerIf.getElseRegion().front());
    Value materialized =
        materializeStructuredLoad(op, ptr, tensorType, loc, rewriter);
    rewriter.create<scf::YieldOp>(loc, materialized);

    rewriter.setInsertionPointAfter(innerIf);
    rewriter.create<scf::YieldOp>(loc, innerIf.getResult(0));

    rewriter.setInsertionPointAfter(outerIf);
    return outerIf.getResult(0);
  }

  Value lowerStructuredLoadWithAliasMemref(tts::LoadOp op, Value ptr,
                                           RankedTensorType tensorType,
                                           MemRefType resultType, Location loc,
                                           ConversionPatternRewriter &rewriter)
      const {
    auto wrapCast = getWrapCast(ptr);
    if (!wrapCast)
      return createAliasedLoadMemref(ptr, tensorType, resultType, loc,
                                     rewriter);

    if (wrapCast->hasAttr(WRAP_LINEAR))
      return materializeStructuredLoadMemref(op, ptr, tensorType, resultType,
                                             loc, rewriter);

    auto memrefs = wrapCast.getOperands();
    assert(memrefs.size() == 2 && "expected two memrefs for split wraparound");
    Value block1 = memrefs[0];
    Value block2 = memrefs[1];

    unsigned splitDim = wrapCast->hasAttr(WRAP_SIDE_BY_SIDE) ? 1 : 0;
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value block1Extent = rewriter.create<memref::DimOp>(loc, block1, splitDim);
    Value block2Extent = rewriter.create<memref::DimOp>(loc, block2, splitDim);
    Value block1Empty = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, block1Extent, zero);
    Value block2Empty = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, block2Extent, zero);

    OpBuilder::InsertionGuard guard(rewriter);
    auto outerIf =
        rewriter.create<scf::IfOp>(loc, TypeRange{resultType}, block2Empty,
                                   /*withElseRegion=*/true);

    rewriter.setInsertionPointToStart(&outerIf.getThenRegion().front());
    Value block1Memref =
        createAliasedLoadMemref(block1, tensorType, resultType, loc, rewriter);
    rewriter.create<scf::YieldOp>(loc, block1Memref);

    rewriter.setInsertionPointToStart(&outerIf.getElseRegion().front());
    auto innerIf =
        rewriter.create<scf::IfOp>(loc, TypeRange{resultType}, block1Empty,
                                   /*withElseRegion=*/true);

    rewriter.setInsertionPointToStart(&innerIf.getThenRegion().front());
    Value block2Memref =
        createAliasedLoadMemref(block2, tensorType, resultType, loc, rewriter);
    rewriter.create<scf::YieldOp>(loc, block2Memref);

    rewriter.setInsertionPointToStart(&innerIf.getElseRegion().front());
    Value materialized = materializeStructuredLoadMemref(
        op, ptr, tensorType, resultType, loc, rewriter);
    rewriter.create<scf::YieldOp>(loc, materialized);

    rewriter.setInsertionPointAfter(innerIf);
    rewriter.create<scf::YieldOp>(loc, innerIf.getResult(0));

    rewriter.setInsertionPointAfter(outerIf);
    return outerIf.getResult(0);
  }

  LogicalResult
  rewriteStructuredLoad(tts::LoadOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    assert(!op.hasMask());

    auto loc = op->getLoc();
    auto ptr = adaptor.getPtr();
    auto tensorType = cast<RankedTensorType>(op.getType());
    assert(!op.getOther() && "other value used in non-masked load");

    if (!enableAliasFirst) {
      Value tensor =
          materializeStructuredLoad(op, ptr, tensorType, loc, rewriter);
      rewriter.replaceOp(op, tensor);
      return success();
    }

    Value tensor;
    auto wrapCast = getWrapCast(ptr);
    if (wrapCast && wrapCast->hasAttr(WRAP_LINEAR)) {
      tensor = materializeStructuredLoad(op, ptr, tensorType, loc, rewriter);
    } else if (wrapCast && (wrapCast->hasAttr(WRAP_SIDE_BY_SIDE) ||
                            wrapCast->hasAttr(WRAP_STACKED))) {
      MemRefType resultType = getLoadResultMemrefType(ptr, tensorType, rewriter);
      Value memref = lowerStructuredLoadWithAliasMemref(
          op, ptr, tensorType, resultType, loc, rewriter);
      tensor = createLoadTensor(memref, tensorType, loc, rewriter,
                                false /* writable */);
    } else {
      tensor = lowerStructuredLoadWithAlias(op, ptr, tensorType, loc, rewriter);
    }
    rewriter.replaceOp(op, tensor);

    return success();
  }

  LogicalResult rewriteMaskedLoad(tts::LoadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    assert(op.hasMask());

    auto loc = op->getLoc();
    auto ptr = adaptor.getPtr();

    auto tensorType = cast<RankedTensorType>(op.getType());
    SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();

    if (!enableAliasFirst) {
      Value tensor = materializeMaskedStructuredLoad(
          op, ptr, tensorType, mixedDims, loc, rewriter);
      rewriter.replaceOp(op, tensor);
      return success();
    }

    MemRefType resultType = getLoadResultMemrefType(ptr, tensorType, rewriter);

    if (auto wrapCast = getWrapCast(ptr);
        wrapCast && wrapCast->hasAttr(WRAP_LINEAR)) {
      Value memref = materializeMaskedStructuredLoadMemref(
          op, ptr, tensorType, mixedDims, resultType, loc, rewriter);
      Value tensor =
          createLoadTensor(memref, tensorType, loc, rewriter,
                           false /* writable */);
      rewriter.replaceOp(op, tensor);
      return success();
    }

    Value fullTile = createFullTileCondition(loc, tensorType.getShape(),
                                             mixedDims, op.getStaticMaskDims(),
                                             rewriter);

    Value memref;
    if (auto fullTileConstant = getConstantBoolValue(fullTile);
        succeeded(fullTileConstant)) {
      memref = *fullTileConstant
                   ? lowerStructuredLoadWithAliasMemref(
                         op, ptr, tensorType, resultType, loc, rewriter)
                   : materializeMaskedStructuredLoadMemref(
                         op, ptr, tensorType, mixedDims, resultType, loc,
                         rewriter);
    } else {
      OpBuilder::InsertionGuard guard(rewriter);
      auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{resultType},
                                             fullTile, /*withElseRegion=*/true);

      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      Value aliasMemref = lowerStructuredLoadWithAliasMemref(
          op, ptr, tensorType, resultType, loc, rewriter);
      rewriter.create<scf::YieldOp>(loc, aliasMemref);

      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      Value materialized = materializeMaskedStructuredLoadMemref(
          op, ptr, tensorType, mixedDims, resultType, loc, rewriter);
      rewriter.create<scf::YieldOp>(loc, materialized);

      rewriter.setInsertionPointAfter(ifOp);
      memref = ifOp.getResult(0);
    }

    Value tensor = createLoadTensor(memref, tensorType, loc, rewriter,
                                    false /* writable */);
    rewriter.replaceOp(op, tensor);

    return success();
  }

  LogicalResult rewriteGather(tts::MakeGatherScatterTensorPtrOp ptr,
                              tts::LoadOp op, Value memRefPtr,
                              ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();

    Value gatherOffset = ptr.getGatherScatterOffset();
    // Cast gatherOffset to index
    auto offsetShapedType = cast<ShapedType>(gatherOffset.getType());
    unsigned offsetSize = offsetShapedType.getShape()[0];
    auto indexOffsetTy = RankedTensorType::get(offsetShapedType.getShape(),
                                               rewriter.getIndexType());
    gatherOffset =
        rewriter.create<arith::IndexCastOp>(loc, indexOffsetTy, gatherOffset)
            .getResult();

    int gatherDim = ptr.getGatherScatterDim();

    auto offsets = ptr.getMixedOffsets();
    auto strides = ptr.getMixedStrides();

    std::vector<int64_t> staticSizes = ptr.getSizes();
    staticSizes[gatherDim] = 1;
    SmallVector<Value> dynSizes; // sizes are always static
    auto sizes = mlir::getMixedValues(staticSizes, dynSizes, rewriter);

    // Create alloc to save the result.
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    auto allocType =
        MemRefType::get(resultType.getShape(), resultType.getElementType());
    auto alloc = rewriter.create<memref::AllocOp>(loc, allocType);

    auto allocStrides = mlir::getMixedValues(
        allocType.getStridesAndOffset().first, dynSizes, rewriter);
    // Fill load destination with other value
    if (Value other = op.getOther()) {
      fillWithValue(loc, alloc, other, resultType.getShape(),
                    op.getMixedMaskDims(), op.getStaticMaskDims(), rewriter);
    }

    // Create loop to iterate every offset in gatherOffset.
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, offsetSize).getResult();
    if (op.hasMask()) {
      SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();
      OpFoldResult gatherMaskDim = mixedDims[gatherDim];
      // If gatherMaskDim is a immediate, we can just update the offsetSize
      // to the value of gatherMaskDim.
      // Otherwise, we will need to compare the induction variable with
      // gatherMaskDim to guard the load.
      if (auto gatherMaskDimIndex = getIntAttr(gatherMaskDim)) {
        // If the gather mask dimension is a constant, we can use it directly.
        unsigned gatherMaskDimValue = gatherMaskDimIndex.value();
        if (gatherMaskDimValue == 0 && ptr.getGatherScatterMask()) {
          // For unstructured mask case, loop over all elements and use the
          // unstructured mask to guard the store.
          gatherMaskDimValue = offsetSize;
        }
        offsetSize = std::min(offsetSize, gatherMaskDimValue);
        upperBound = rewriter.create<arith::ConstantIndexOp>(loc, offsetSize)
                         .getResult();
      } else {
        // Use arith::MinSIOp to get the minimum value of gatherMaskDim
        // and offsetSize.
        auto gatherMaskDimVal = cast<Value>(gatherMaskDim);
        auto offsetSizeVal =
            rewriter.create<arith::ConstantIndexOp>(loc, offsetSize);
        upperBound =
            rewriter
                .create<arith::MinSIOp>(loc, gatherMaskDimVal, offsetSizeVal)
                .getResult();
      }
    }
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

    // Create tensor from alloc and use it as the result to replace op.
    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, op.getType(), alloc, true /* restrict */, true /* writable */);
    rewriter.replaceOp(op, tensor);

    // Build loop body.
    rewriter.setInsertionPointToStart(loop.getBody());

    Value inductionVar = loop.getInductionVar();

    if (Value unstructuredMask = ptr.getGatherScatterMask()) {
      // If the gather scatter mask is present, we need to use it to guard the
      // load.
      auto maskValue = rewriter.create<tensor::ExtractOp>(
          loc, unstructuredMask, ValueRange{inductionVar});
      auto ifOp = rewriter.create<scf::IfOp>(loc, maskValue);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    }

    // Load the offsetElt first.
    auto gatherOffsetElt = rewriter.create<tensor::ExtractOp>(
        loc, gatherOffset, ValueRange{inductionVar});

    // reinterpret_cast to current row as memRefPtr[gatherOffsetElt].
    Value srcPtr = rewriteGatherScatterPtrElement(staticSizes, ptr, memRefPtr,
                                                  gatherOffsetElt.getResult(),
                                                  gatherDim, rewriter);
    unsigned rank = ptr.getSizes().size();
    // The subview should not apply an additional stride to the source.
    SmallVector<OpFoldResult> oneStrides(rank, OpFoldResult(step));
    // subview from srcPtr for mask.
    // With offsets[gatherDim] set to 0 since the offset already in
    // reinterpret_cast. With sizes[gatherDim] set to 1 since we are load one
    // row each time.
    if (op.hasMask()) {
      SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();
      mixedDims[gatherDim] = sizes[gatherDim];
      sizes = mixedDims;
      // maskOffsets should be all zero, since srcPtr already has the offsets.
      SmallVector<OpFoldResult> maskOffsets(rank, OpFoldResult(lowerBound));
      // Use oneStrides for subview.
      auto dstSubViewType = memref::SubViewOp::inferResultType(
          cast<MemRefType>(srcPtr.getType()), maskOffsets, sizes, oneStrides);
      srcPtr =
          rewriter
              .create<memref::SubViewOp>(loc, cast<MemRefType>(dstSubViewType),
                                         srcPtr, maskOffsets, sizes, oneStrides)
              .getResult();
    }

    // alloc[inductionVar]
    SmallVector<OpFoldResult> allocOffsets(rank, OpFoldResult(lowerBound));
    allocOffsets[gatherDim] = inductionVar;
    auto dstAllocType = memref::SubViewOp::inferResultType(
        allocType, allocOffsets, sizes, oneStrides);
    auto dstSubview = rewriter.create<memref::SubViewOp>(
        loc, cast<MemRefType>(dstAllocType), alloc, allocOffsets, sizes,
        oneStrides);
    // Copy srcPtr to alloc[inductionVar].
    rewriter.create<memref::CopyOp>(loc, srcPtr, dstSubview);

    return success();
  }

public:
  LoadConverter(const TypeConverter &typeConverter, MLIRContext *context,
                bool enableAliasFirst)
      : OpConversionPattern<tts::LoadOp>(typeConverter, context),
        enableAliasFirst(enableAliasFirst) {}

  LogicalResult
  matchAndRewrite(tts::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto gatherScatterPtr = getGatherScatterPtrDef(op.getPtr())) {
      return rewriteGather(gatherScatterPtr, op, adaptor.getPtr(), rewriter);
    }

    if (op.hasMask()) {
      return rewriteMaskedLoad(op, adaptor, rewriter);
    } else {
      return rewriteStructuredLoad(op, adaptor, rewriter);
    }
  }
};

struct StoreConverter : public OpConversionPattern<tts::StoreOp> {
private:
  using OpConversionPattern<tts::StoreOp>::OpConversionPattern;

  static tensor::ExtractSliceOp
  getExtractSlice(int rank, ArrayRef<OpFoldResult> dims, Value source,
                  const Location loc, OpBuilder &b) {
    auto sourceType = cast<RankedTensorType>(source.getType());
    SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));

    auto dstType = tensor::ExtractSliceOp::inferResultType(sourceType, offsets,
                                                           dims, strides);

    return b.create<tensor::ExtractSliceOp>(loc, dstType, source, offsets, dims,
                                            strides);
  }

  LogicalResult rewriteScatter(tts::MakeGatherScatterTensorPtrOp ptr,
                               tts::StoreOp op, Value memRefPtr, Value stVal,
                               ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();

    Value gatherOffset = ptr.getGatherScatterOffset();
    // Cast gatherOffset to index.
    auto offsetShapedType = cast<ShapedType>(gatherOffset.getType());
    unsigned offsetSize = offsetShapedType.getShape()[0];
    auto indexOffsetTy = RankedTensorType::get(offsetShapedType.getShape(),
                                               rewriter.getIndexType());
    gatherOffset =
        rewriter.create<arith::IndexCastOp>(loc, indexOffsetTy, gatherOffset)
            .getResult();

    int gatherDim = ptr.getGatherScatterDim();

    auto offsets = ptr.getMixedOffsets();
    auto strides = ptr.getMixedStrides();

    std::vector<int64_t> staticSizes = ptr.getSizes();
    staticSizes[gatherDim] = 1;
    SmallVector<Value> dynSizes; // sizes are always static
    auto sizes = mlir::getMixedValues(staticSizes, dynSizes, rewriter);

    // Create loop to iterate every offset in gatherOffset.
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, offsetSize).getResult();
    if (op.hasMask()) {
      SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();
      OpFoldResult gatherMaskDim = mixedDims[gatherDim];
      // If gatherMaskDim is a immediate, we can just update the offsetSize
      // to the value of gatherMaskDim.
      // Otherwise, we will need to compare the induction variable with
      // gatherMaskDim to guard the load.
      if (auto gatherMaskDimIndex = getIntAttr(gatherMaskDim)) {
        // If the gather mask dimension is a constant, we can use it directly.
        unsigned gatherMaskDimValue = gatherMaskDimIndex.value();
        if (gatherMaskDimValue == 0 && ptr.getGatherScatterMask()) {
          // For unstructured mask case, loop over all elements and use the
          // unstructured mask to guard the store.
          gatherMaskDimValue = offsetSize;
        }
        offsetSize = std::min(offsetSize, gatherMaskDimValue);
        upperBound = rewriter.create<arith::ConstantIndexOp>(loc, offsetSize)
                         .getResult();
      } else {
        // Use arith::MinSIOp to get the minimum value of gatherMaskDim
        // and offsetSize.
        auto gatherMaskDimVal = cast<Value>(gatherMaskDim);
        auto offsetSizeVal =
            rewriter.create<arith::ConstantIndexOp>(loc, offsetSize);
        upperBound =
            rewriter
                .create<arith::MinSIOp>(loc, gatherMaskDimVal, offsetSizeVal)
                .getResult();
      }
    }
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

    // Build loop body.
    rewriter.setInsertionPointToStart(loop.getBody());

    Value inductionVar = loop.getInductionVar();

    if (Value unstructuredMask = ptr.getGatherScatterMask()) {
      // If the gather scatter mask is present, we need to use it to guard the
      // store.
      auto maskValue = rewriter.create<tensor::ExtractOp>(
          loc, unstructuredMask, ValueRange{inductionVar});
      auto ifOp = rewriter.create<scf::IfOp>(loc, maskValue);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    }

    // Load the offsetElt first.
    auto gatherOffsetElt = rewriter.create<tensor::ExtractOp>(
        loc, gatherOffset, ValueRange{inductionVar});

    // Create extract_slice stVal[inductionVar].
    unsigned rank = ptr.getSizes().size();
    SmallVector<OpFoldResult> stValOffsets(rank, OpFoldResult(lowerBound));
    stValOffsets[gatherDim] = inductionVar;

    // Use mixed mask dims as sizes with mixedDims[gatherDim] set to 1 when
    // hasMask.
    if (op.hasMask()) {
      SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();
      mixedDims[gatherDim] = sizes[gatherDim];
      sizes = mixedDims;
    }
    // The subview should not apply an additional stride to the source.
    SmallVector<OpFoldResult> oneStrides(rank, OpFoldResult(step));
    auto slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, stVal, stValOffsets, sizes, oneStrides);

    // reinterpret_cast to current row as memRefPtr[gatherOffsetElt].
    Value dstPtr = rewriteGatherScatterPtrElement(staticSizes, ptr, memRefPtr,
                                                  gatherOffsetElt.getResult(),
                                                  gatherDim, rewriter);
    // subview from dstPtr for mask.
    // Set offsets[] to 0 since it gatherOffsetElt already in reinterpret_cast.
    if (op.hasMask()) {
      // maskOffsets should be all zero, since srcPtr already has the offsets.
      SmallVector<OpFoldResult> maskOffsets(rank, OpFoldResult(lowerBound));
      auto dstType = memref::SubViewOp::inferResultType(
          cast<MemRefType>(dstPtr.getType()), maskOffsets, sizes, oneStrides);

      dstPtr =
          rewriter
              .create<memref::SubViewOp>(loc, cast<MemRefType>(dstType), dstPtr,
                                         maskOffsets, sizes, oneStrides)
              .getResult();
    }
    // store slice to dstPtr.
    auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
        loc, slice, dstPtr);
    storeOp.setWritable(true);

    rewriter.eraseOp(op);

    return success();
  }

public:
  StoreConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::StoreOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    if (auto gatherScatterPtr = getGatherScatterPtrDef(op.getPtr())) {
      return rewriteScatter(gatherScatterPtr, op, adaptor.getPtr(),
                            adaptor.getValue(), rewriter);
    }

    auto ptr = unwrapMemrefIdentityCasts(adaptor.getPtr());
    auto storeValue = op.getValue();
    auto rank = cast<RankedTensorType>(storeValue.getType()).getRank();

    if (!isa<MemRefType>(ptr.getType())) {
      return rewriter.notifyMatchFailure(
          op, "expected memref pointer in store lowering");
    }

    if (op.hasMask()) {
      auto mixedDims = op.getMixedMaskDims();

      auto srcSlice =
          getExtractSlice(rank, mixedDims, storeValue, loc, rewriter);
      auto dstSubview = getSubview(rank, mixedDims, ptr, loc, rewriter);

      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, srcSlice, dstSubview);
      storeOp.setWritable(true);
    } else {
      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, storeValue, ptr);
      storeOp.setWritable(true);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::populateStructuredToMemrefConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    bool enableAliasFirst) {
  patterns.add<MakeTensorPtrConverter, MakeGatherScatterTensorPtrConverter>(
      typeConverter, patterns.getContext());
  patterns.add<LoadConverter>(typeConverter, patterns.getContext(),
                              enableAliasFirst);
  patterns.add<StoreConverter>(typeConverter, patterns.getContext());
}

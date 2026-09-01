//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Types.h"

#define DEBUG_TYPE "structured-to-memref"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_STRUCTUREDTOMEMREF
#include "triton-shared/Conversion/StructuredToMemref/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

static constexpr StringLiteral kWrapSideBySideAttr = "wrap_side_by_side";
static constexpr StringLiteral kWrapStackedAttr = "wrap_stacked";
static constexpr StringLiteral kWrapLinearAttr = "wrap_linear";

static MemRefType getDynamicStridedMemRefType(RankedTensorType tensorType,
                                              Type elementType) {
  SmallVector<int64_t> dynamicStrides(tensorType.getRank(),
                                      ShapedType::kDynamic);
  auto layout = StridedLayoutAttr::get(tensorType.getContext(),
                                       ShapedType::kDynamic, dynamicStrides);
  return MemRefType::get(tensorType.getShape(), elementType, layout);
}

static bool hasWrapAttr(UnrealizedConversionCastOp op) {
  return op->hasAttr(kWrapSideBySideAttr) || op->hasAttr(kWrapStackedAttr) ||
         op->hasAttr(kWrapLinearAttr);
}

static bool hasAutomaticAllocationScopeAncestor(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent != nullptr;
       parent = parent->getParentOp()) {
    if (parent->hasTrait<OpTrait::AutomaticAllocationScope>())
      return true;
  }
  return false;
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
           "expected static memref shape while resolving wrap cast");
    count *= extent;
  }
  return count;
}

static Value createLayoutAwareTemporary(MemRefType resultType, Location loc,
                                        Operation *anchor,
                                        IRRewriter &rewriter) {
  ArrayRef<int64_t> shape = resultType.getShape();
  SmallVector<int64_t> staticStrides =
      getLayoutClassStaticStrides(shape, resultType);
  SmallVector<int64_t> compactStrides =
      getCompactStrideValues(shape, staticStrides);
  int64_t elementCount = getStaticElementCount(shape);

  auto backingType = MemRefType::get({elementCount}, resultType.getElementType(),
                                     AffineMap(), resultType.getMemorySpace());
  Value backing;
  if (hasAutomaticAllocationScopeAncestor(anchor))
    backing = rewriter.create<memref::AllocaOp>(loc, backingType);
  else
    backing = rewriter.create<memref::AllocOp>(loc, backingType);

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
      .create<memref::ReinterpretCastOp>(loc, resultType, backing,
                                         rewriter.getIndexAttr(0), sizes,
                                         strides)
      .getResult();
}

static void createSideBySideCopies(Value block1, Value block2, Value dst,
                                   Location loc, IRRewriter &rewriter) {
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
  Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);
  Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
  Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

  auto block1Dst =
      rewriter.create<memref::SubViewOp>(loc, dst, ValueRange{zero, zero},
                                         ValueRange{block1Row, block1Col},
                                         ValueRange{one, one});
  auto block2Dst = rewriter.create<memref::SubViewOp>(
      loc, dst, ValueRange{zero, block1Col}, ValueRange{block2Row, block2Col},
      ValueRange{one, one});

  rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
  rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
}

static void createStackedCopies(Value block1, Value block2, Value dst,
                                Location loc, IRRewriter &rewriter) {
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
  Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);
  Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
  Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

  auto block1Dst =
      rewriter.create<memref::SubViewOp>(loc, dst, ValueRange{zero, zero},
                                         ValueRange{block1Row, block1Col},
                                         ValueRange{one, one});
  auto block2Dst = rewriter.create<memref::SubViewOp>(
      loc, dst, ValueRange{block1Row, zero}, ValueRange{block2Row, block2Col},
      ValueRange{one, one});

  rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
  rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
}

static void createLinearWrapCopy(UnrealizedConversionCastOp wrapCast, Value dst,
                                 Value copyLimit, Location loc,
                                 IRRewriter &rewriter) {
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
  Value linearBase =
      rewriter
          .create<memref::ReinterpretCastOp>(loc, linearType, basePtr, zero,
                                             ValueRange{wrapBound},
                                             ValueRange{one})
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

static LogicalResult resolveWrapCast(UnrealizedConversionCastOp wrapCast,
                                     IRRewriter &rewriter) {
  if (!hasWrapAttr(wrapCast))
    return failure();
  if (wrapCast->getNumResults() != 1)
    return failure();

  auto resultType = dyn_cast<MemRefType>(wrapCast.getResult(0).getType());
  if (!resultType)
    return wrapCast.emitError("expected memref result type on surviving wrap cast");

  Location loc = wrapCast.getLoc();
  rewriter.setInsertionPoint(wrapCast);

  if (wrapCast->hasAttr(kWrapLinearAttr)) {
    Value concrete = createLayoutAwareTemporary(resultType, loc, wrapCast, rewriter);
    Value copyLimit = rewriter.create<memref::DimOp>(loc, concrete, 0);
    createLinearWrapCopy(wrapCast, concrete, copyLimit, loc, rewriter);
    rewriter.replaceOp(wrapCast, concrete);
    return success();
  }

  if (wrapCast.getNumOperands() != 2)
    return wrapCast.emitError("expected two operands for split wrap cast");

  Value block1 = wrapCast.getOperand(0);
  Value block2 = wrapCast.getOperand(1);
  unsigned splitDim = wrapCast->hasAttr(kWrapSideBySideAttr) ? 1 : 0;

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value block1Extent = rewriter.create<memref::DimOp>(loc, block1, splitDim);
  Value block2Extent = rewriter.create<memref::DimOp>(loc, block2, splitDim);
  Value block1Empty = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, block1Extent, zero);
  Value block2Empty = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, block2Extent, zero);

  auto outerIf = rewriter.create<scf::IfOp>(
      loc, TypeRange{resultType}, block2Empty, /*withElseRegion=*/true);

  rewriter.setInsertionPointToStart(&outerIf.getThenRegion().front());
  Value block1View = rewriter.create<memref::CastOp>(loc, resultType, block1);
  rewriter.create<scf::YieldOp>(loc, block1View);

  rewriter.setInsertionPointToStart(&outerIf.getElseRegion().front());
  auto innerIf = rewriter.create<scf::IfOp>(
      loc, TypeRange{resultType}, block1Empty, /*withElseRegion=*/true);

  rewriter.setInsertionPointToStart(&innerIf.getThenRegion().front());
  Value block2View = rewriter.create<memref::CastOp>(loc, resultType, block2);
  rewriter.create<scf::YieldOp>(loc, block2View);

  rewriter.setInsertionPointToStart(&innerIf.getElseRegion().front());
  Value materialized =
      createLayoutAwareTemporary(resultType, loc, wrapCast, rewriter);
  if (wrapCast->hasAttr(kWrapSideBySideAttr))
    createSideBySideCopies(block1, block2, materialized, loc, rewriter);
  else
    createStackedCopies(block1, block2, materialized, loc, rewriter);
  rewriter.create<scf::YieldOp>(loc, materialized);

  rewriter.setInsertionPointAfter(innerIf);
  rewriter.create<scf::YieldOp>(loc, innerIf.getResult(0));

  rewriter.replaceOp(wrapCast, outerIf.getResults());
  return success();
}

class PtrToUnrankedMemrefConverter : public TypeConverter {
public:
  PtrToUnrankedMemrefConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      Type pointeeType = ptrType.getPointeeType();
      if (auto tensorType = dyn_cast<TensorType>(pointeeType))
        pointeeType = tensorType.getElementType();
      return UnrankedMemRefType::get(pointeeType, 0);
    });
    addConversion([](RankedTensorType tensorType) -> Type {
      auto ptrType = dyn_cast<triton::PointerType>(tensorType.getElementType());
      if (!ptrType)
        return tensorType;
      return getDynamicStridedMemRefType(tensorType, ptrType.getPointeeType());
    });
    addTargetMaterialization([&](OpBuilder &builder,
                                 UnrankedMemRefType resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
    addTargetMaterialization([&](OpBuilder &builder, MemRefType resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  }
};

class StructuredToMemrefPass
    : public triton::impl::StructuredToMemrefBase<StructuredToMemrefPass> {
  using StructuredToMemrefBase<StructuredToMemrefPass>::StructuredToMemrefBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<tptr::TPtrDialect, func::FuncDialect, arith::ArithDialect,
                math::MathDialect, linalg::LinalgDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect,
                bufferization::BufferizationDialect, triton::TritonDialect,
                ttx::TritonTilingExtDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        memref::MemRefDialect>();

    target.addIllegalOp<tts::LoadOp, tts::StoreOp, tts::MakeTensorPtrOp>();
    // target.addLegalOp<tts::MakeGatherScatterTensorPtrOp>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    PtrToUnrankedMemrefConverter typeConverter;

    triton::populateStructuredToMemrefConversionPatterns(
        patterns, typeConverter, enableAliasFirst);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    IRRewriter rewriter(&getContext());
    SmallVector<UnrealizedConversionCastOp> wrapCasts;
    moduleOp.walk([&](UnrealizedConversionCastOp op) {
      if (hasWrapAttr(op))
        wrapCasts.push_back(op);
    });
    for (UnrealizedConversionCastOp wrapCast : wrapCasts) {
      if (!wrapCast->getBlock())
        continue;
      if (failed(resolveWrapCast(wrapCast, rewriter))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<Operation *> deadGatherScatterPtrs;
    moduleOp.walk([&](tts::MakeGatherScatterTensorPtrOp op) {
      if (op->use_empty()) {
        deadGatherScatterPtrs.push_back(op.getOperation());
      }
    });
    for (Operation *op : deadGatherScatterPtrs) {
      op->erase();
    }

    // Some conversion paths materialize temporary unrealized casts that become
    // dead after load/store rewrites. Remove them transitively so ptr-typed
    // tensor results do not leak into later stages.
    while (true) {
      SmallVector<Operation *> deadCasts;
      moduleOp.walk([&](UnrealizedConversionCastOp op) {
        if (op->use_empty()) {
          deadCasts.push_back(op.getOperation());
        }
      });
      if (deadCasts.empty()) {
        break;
      }
      for (Operation *op : deadCasts) {
        op->erase();
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createStructuredToMemrefPass(bool enableAliasFirst) {
  StructuredToMemrefOptions options;
  options.enableAliasFirst = enableAliasFirst;
  return std::make_unique<StructuredToMemrefPass>(options);
}

//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Analysis/MaskAnalysis.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>
#include <cstddef>
#include <optional>
#include <queue>
#include <string>

#define DEBUG_TYPE "triton-ptr-analysis"

using namespace mlir;


static Value applyUnstructuredMask(Operation *op, Value ptr,
                                   triton::MaskState &mstate, Location loc,
                                   OpBuilder builder) {
  SmallVector<std::pair<unsigned, Value>> masks = mstate.getUnstructuredMasks();
  if (masks.empty()) {
    return ptr;
  }
  if (masks.size() > 1) {
    LLVM_DEBUG(op->emitRemark(
        "MaskAnalysis failed for more than one unstructured masks"));
    return nullptr;
  }

  auto [dim, unstructuredMask] = masks[0];
  if (auto gatherScatterPtr =
          ptr.getDefiningOp<tts::MakeGatherScatterTensorPtrOp>()) {
    if (dim != gatherScatterPtr.getGatherScatterDim()) {
      LLVM_DEBUG(op->emitRemark(
          "MaskAnalysis failed for unstructured mask dim not equal "
          "gather scatter dim"));
      return nullptr;
    }

    ptr =
        builder
            .create<tts::MakeGatherScatterTensorPtrOp>(
                loc, gatherScatterPtr.getBase(),
                gatherScatterPtr.getGatherScatterOffset(), unstructuredMask,
                gatherScatterPtr.getGatherScatterDim(),
                gatherScatterPtr.getSizes(), gatherScatterPtr.getMixedStrides(),
                gatherScatterPtr.getMixedOffsets())
            .getResult();
  } else if (auto structuredPtr = ptr.getDefiningOp<tts::MakeTensorPtrOp>()) {
    auto ofrToI32Value = [&](OpFoldResult ofr) {
      Value v = dyn_cast<Value>(ofr);
      if (!v) {
        v = builder
                .create<arith::ConstantOp>(
                    loc, cast<TypedAttr>(cast<Attribute>(ofr)))
                .getResult();
      }
      if (isa<IndexType>(v.getType())) {
        v = builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), v)
                .getResult();
      } else if (v.getType().isInteger(64)) {
        v = builder.create<arith::TruncIOp>(loc, builder.getI32Type(), v)
                .getResult();
      }

      return v;
    };
    OpFoldResult offsetFold = structuredPtr.getMixedOffsets()[dim];
    Value offset = ofrToI32Value(offsetFold);
    auto offsetRowType = RankedTensorType::get({structuredPtr.getSizes()[dim]},
                                               offset.getType());
    OpFoldResult strideFold = structuredPtr.getMixedStrides()[dim];
    Value stride = ofrToI32Value(strideFold);
    
    
    
    offset = builder.create<arith::DivUIOp>(loc, offset, stride);

    Value gatherScatterOffset =
        builder.create<tensor::SplatOp>(loc, offsetRowType, offset).getResult();
    Value range = builder
                      .create<triton::MakeRangeOp>(
                          loc, offsetRowType, 0, structuredPtr.getSizes()[dim])
                      .getResult();
    gatherScatterOffset =
        builder.create<arith::AddIOp>(loc, gatherScatterOffset, range);
    ptr = builder
              .create<tts::MakeGatherScatterTensorPtrOp>(
                  loc, structuredPtr.getBase(), gatherScatterOffset,
                  unstructuredMask, dim, structuredPtr.getSizes(),
                  structuredPtr.getMixedStrides(),
                  structuredPtr.getMixedOffsets())
              .getResult();
  } else {
    return nullptr;
  }
  
  mstate.dims[dim] = OpFoldResult(builder.getI32IntegerAttr(0));
  return ptr;
}

static Type getStorageElementType(Type elemType) {
  if (elemType.isInteger(1)) {
    return IntegerType::get(elemType.getContext(), 8);
  }
  return elemType;
}

static std::optional<Type> getTensorPtrPointeeElementType(Type type) {
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType) {
    return std::nullopt;
  }

  auto ptrType = dyn_cast<triton::PointerType>(shapedType.getElementType());
  if (!ptrType) {
    return std::nullopt;
  }

  return ptrType.getPointeeType();
}

static std::optional<unsigned> getStorageElementBitWidth(Type elemType) {
  auto storageElemType = getStorageElementType(elemType);
  if (!storageElemType.isIntOrFloat()) {
    return std::nullopt;
  }
  return storageElemType.getIntOrFloatBitWidth();
}

static bool isStorageCompatibleTensorPtrBitcast(triton::BitcastOp op) {
  auto srcElemType = getTensorPtrPointeeElementType(op.getSrc().getType());
  auto dstElemType = getTensorPtrPointeeElementType(op.getType());
  if (!srcElemType || !dstElemType) {
    return false;
  }

  auto srcWidth = getStorageElementBitWidth(*srcElemType);
  auto dstWidth = getStorageElementBitWidth(*dstElemType);
  return srcWidth && dstWidth && *srcWidth == *dstWidth;
}

static FailureOr<Value> createBitcastedScalarBasePointer(Value source,
                                                         Type resultType,
                                                         Location loc,
                                                         OpBuilder &builder) {
  auto srcPtrType = dyn_cast<triton::PointerType>(source.getType());
  if (!srcPtrType) {
    return failure();
  }

  auto dstElemType = getTensorPtrPointeeElementType(resultType);
  if (!dstElemType) {
    return failure();
  }

  auto dstPtrType =
      triton::PointerType::get(*dstElemType, srcPtrType.getAddressSpace());
  if (dstPtrType == source.getType()) {
    return source;
  }

  return builder.create<triton::BitcastOp>(loc, dstPtrType, source)
      .getResult();
}

namespace mlir {

namespace tts {

int32_t PtrState::getRank() const {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size() &&
         shape.size() == offsets.size());
  return offsets.size();
}

bool PtrState::isEmpty() const {
  return (getRank() == 0 && !source && !scalar);
}

bool PtrState::hasModulo() const {
  for (int32_t i = 0; i < getRank(); i++) {
    if (dimHasModulo(i)) {
      return true;
    }
  }
  return false;
}

bool PtrState::dimHasModulo(uint32_t dim) const {
  assert(
      !isBlockPtr() &&
      "Analysis should not check modulo if PtrState describes block pointer");

  assert(dim < getRank());

  auto intAttr = getIntAttr(shape[dim]);
  if (!intAttr.has_value()) {
    return true;
  }

  return intAttr.value() != 0;
}

bool isNotStructured(OpFoldResult offset) {
  auto value = dyn_cast<Value>(offset);
  return value && isa<ShapedType>(value.getType());
}

bool PtrState::dimIsStructured(uint32_t dim) const {
  assert(dim < getRank());

  return !isNotStructured(offsets[dim]);
}

int32_t PtrState::getNonStructuredDim() const {
  SmallVector<int32_t> dims;
  for (int32_t i = 0; i < getRank(); i++) {
    if (dimIsStructured(i))
      continue;
    dims.emplace_back(i);
  }
  assert(dims.size() == 1 && "must have single non-continuous dimension");
  return dims.front();
}

bool PtrState::noStructuredDimExists() const {
  return getRank() > 0 && llvm::all_of(offsets, [](OpFoldResult offset) {
           return isNotStructured(offset);
         });
}

bool PtrState::isStructured() const {
  return llvm::all_of(
      offsets, [](OpFoldResult offset) { return !isNotStructured(offset); });
}

bool PtrState::isBlockPtr() const { return !order.empty(); }

bool isNotSingleDim(Value v) {
  auto shapedTy = dyn_cast<ShapedType>(v.getType());
  if (!shapedTy)
    return false;
  auto valShape = shapedTy.getShape();

  
  return llvm::find_singleton<int64_t>(
             valShape,
             [](int64_t size, bool) {
               return size > 1 ? (int64_t *)size : nullptr;
             },
             false) == nullptr;
}

LogicalResult PtrState::rebuildAsUnsupportedOp(Value operand) {
  if (isNotSingleDim(operand))
    return failure();

  if (!isEmpty())
    return failure();

  
  
  auto opType = cast<ShapedType>(operand.getType());
  
  
  if (isa<triton::PointerType>(opType.getElementType()))
    return failure();

  auto opShape = opType.getShape();

  
  auto indexTy = IndexType::get(operand.getContext());
  auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
  auto index1 = IntegerAttr::get(indexTy, APInt(64, 1));
  for (auto size : opShape) {
    if (size == 1) {
      offsets.push_back(index0);
      strides.push_back(index0);
    } else {
      offsets.push_back(operand);
      strides.push_back(index1);
    }
    sizes.push_back(IntegerAttr::get(indexTy, APInt(64, size)));
    shape.push_back(index0);
  }
  return success();
}

LogicalResult PtrState::rebuildAsGatherScatter(Value op, int nonContinuousDim) {
  if (isNotSingleDim(op))
    return failure();
  if (nonContinuousDim >= getRank())
    return failure();

  
  
  auto opShape = cast<ShapedType>(op.getType()).getShape();
  
  
  if (opShape[nonContinuousDim] <= 1)
    return failure();

  
  auto indexTy = IndexType::get(op.getContext());
  auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
  auto index1 = IntegerAttr::get(indexTy, APInt(64, 1));

  offsets[nonContinuousDim] = op;
  strides[nonContinuousDim] = index1;
  shape[nonContinuousDim] = index0;
  return success();
}

LogicalResult PtrState::addState(const PtrState &lhsState,
                                 const PtrState &rhsState,
                                 bool isAnalysisingUnstructured, Operation *op,
                                 OpBuilder &builder) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());
  auto loc = op->getLoc();

  if (lhsState.source && rhsState.source) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: do not support adding two pointer states that both "
        "have base pointers"));
    return failure();
  }

  source = lhsState.source ? lhsState.source : rhsState.source;

  if (lhsState.scalar && rhsState.scalar) {
    auto addOp =
        builder.create<arith::AddIOp>(loc, lhsState.scalar, rhsState.scalar);
    scalar = addOp.getResult();
  } else if (lhsState.getRank() == 0) { 
    scalar = lhsState.scalar ? lhsState.scalar : rhsState.scalar;
  }

  if (!lhsState.isStructured() && !rhsState.isStructured()) {
    if (lhsState.getNonStructuredDim() != rhsState.getNonStructuredDim()) {
      LLVM_DEBUG(op->emitRemark(
          "PtrAnalysis: do not support adding two pointer states "
          "that have different non-continuous dimension"));
      return failure();
    }
  }

  for (uint64_t i = 0; i < lhsState.getRank(); i++) {
    if (lhsState.dimIsStructured(i) && rhsState.dimIsStructured(i)) {
      auto newOffset =
          addOFRs(lhsState.offsets[i], rhsState.offsets[i], loc, builder);
      offsets.push_back(newOffset);
      auto newStride =
          addOFRs(lhsState.strides[i], rhsState.strides[i], loc, builder);
      strides.push_back(newStride);
    } else {
      if (isAnalysisingUnstructured) {
        assert(!lhsState.hasModulo() && !rhsState.hasModulo() &&
               "should not have dimension with modulo when analysing "
               "unstructured");
        if (hasConstZero(lhsState.strides[i]) &&
            hasConstZero(lhsState.offsets[i])) {
          
          offsets.push_back(rhsState.offsets[i]);
          strides.push_back(rhsState.strides[i]);
        } else if (hasConstZero(rhsState.strides[i]) &&
                   hasConstZero(rhsState.offsets[i])) {
          
          offsets.push_back(lhsState.offsets[i]);
          strides.push_back(lhsState.strides[i]);
        } else {
          OpFoldResult lhsOffset = lhsState.offsets[i];
          OpFoldResult rhsOffset = rhsState.offsets[i];
          OpFoldResult lhsStride = lhsState.strides[i];
          OpFoldResult rhsStride = rhsState.strides[i];
          
          
          
          if (hasConstZero(lhsStride)) {
            assert(lhsState.dimIsStructured(i) &&
                   !rhsState.dimIsStructured(i) &&
                   "If lhs stride is zero, it must be structured and rhs "
                   "stride is unstructured");
            lhsStride = builder.getIndexAttr(1);
          }
          if (hasConstZero(rhsStride)) {
            assert(rhsState.dimIsStructured(i) &&
                   !lhsState.dimIsStructured(i) &&
                   "If rhs stride is zero, it must be structured and lhs "
                   "stride is unstructured");
            rhsStride = builder.getIndexAttr(1);
          }

          
          
          
          
          
          if (lhsOffset != rhsOffset && lhsStride != rhsStride) {
            
            OpFoldResult stride =
                expandOFRIndex(lhsStride, lhsOffset, loc, builder);
            
            lhsOffset = mulOFRs(lhsOffset, stride, loc, builder);
            
            stride = expandOFRIndex(rhsStride, rhsOffset, loc, builder);
            
            rhsOffset = mulOFRs(rhsOffset, stride, loc, builder);
            
            lhsStride = builder.getIndexAttr(1);
            rhsStride = builder.getIndexAttr(1);
          }

          if (lhsStride == rhsStride) {
            
            
            
            
            
            
            
            if (!lhsState.dimIsStructured(i)) {
              rhsOffset = expandOFRIndex(rhsOffset, lhsOffset, loc, builder);
            } else {
              lhsOffset = expandOFRIndex(lhsOffset, rhsOffset, loc, builder);
            }
            
            offsets.push_back(addOFRs(lhsOffset, rhsOffset, loc, builder));
            
            strides.push_back(lhsStride);
          } else {
            
            
            
            assert(lhsOffset == rhsOffset &&
                   "If strides are not equal, offsets must be equal");
            
            
            
            
            

            
            offsets.push_back(lhsOffset);
            
            strides.push_back(addOFRs(lhsStride, rhsStride, loc, builder));
          }
        }
      } else {
        
        strides.push_back(builder.getIndexAttr(1));
        
        auto newLhsOffset = lhsState.offsets[i];
        auto newRhsOffset = rhsState.offsets[i];
        
        
        
        auto newOffset =
            lhsState.dimIsStructured(i) ? newRhsOffset : newLhsOffset;
        offsets.push_back(newOffset);
      }
    }

    sizes.push_back(lhsState.sizes[i]);
  }

  
  if (lhsState.hasModulo() && rhsState.hasModulo()) {
    LLVM_DEBUG(
        op->emitRemark("PtrAnalysis: do not support adding two pointer states "
                       "that both have modulo"));
    return failure();
  }

  if (lhsState.hasModulo() || rhsState.hasModulo()) {
    
    assert(lhsState.getRank() <= 2);
  }

  
  
  
  
  
  
  
  

  
  
  
  
  
  
  
  

  
  
  
  
  
  PtrState const *lhs = &lhsState;
  PtrState const *rhs = &rhsState;

  if (rhs->hasModulo()) {
    std::swap(lhs, rhs);
  }

  auto indexTy = IndexType::get(op->getContext());
  auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
  for (uint64_t i = 0; i < lhs->getRank(); i++) {
    if (!lhs->dimIsStructured(i) || !rhs->dimIsStructured(i)) {
      
      shape.push_back(index0);
      continue;
    }

    if (!lhs->dimHasModulo(i)) {
      shape.push_back(lhs->shape[i]);
    } else if (hasConstZero(rhs->offsets[i])) {
      shape.push_back(lhs->shape[i]);
    } else if (i == 0 && lhs->getRank() == 2 && rhs->scalar) {
      shape.push_back(lhs->shape[1]);
      shape.push_back(lhs->shape[0]);
      LLVM_DEBUG(op->emitWarning(
          "PtrAnalysis: allowing adding pointer state with modulo in dim 0 to "
          "another pointer state with offset in dim 0.\nPlease verify the "
          "operand that contains a scalar is meant to increment pointers in "
          "dim1. If that is not the case it WILL LEAD TO WRONG COMPILATION "
          "RESULTS.\n\nTo avoid this warning, use expand_dims (instead of "
          "splat) to explicitly specify which dimension contains the scalar."));
      break;
    } else {
      LLVM_DEBUG(op->emitRemark(
          "PtrAnalysis: do not support adding to operand with modulo"));
      return failure();
    }
  }

  return success();
}

void PtrState::dump() const {
  llvm::dbgs() << "PtrState: ";
  if (source) {
    llvm::dbgs() << "source: " << source << "\n";
  }
  if (scalar) {
    llvm::dbgs() << "scalar: " << scalar << "\n";
  }

  llvm::dbgs() << "offsets:\n";
  llvm::interleave(offsets, llvm::dbgs(), "\n");
  llvm::dbgs() << "\nstrides:\n";
  llvm::interleave(strides, llvm::dbgs(), "\n");
  llvm::dbgs() << "\nsizes:\n";
  llvm::interleave(sizes, llvm::dbgs(), "\n");
  llvm::dbgs() << "\nshape:\n";
  llvm::interleave(shape, llvm::dbgs(), "\n");
  llvm::dbgs() << "\norder:\n";
  llvm::interleave(order, llvm::dbgs(), "\n");
  if (isStructured()) {
    llvm::dbgs() << "structured\n";
  } else {
    for (int i = 0; i < getRank(); i++) {
      llvm::dbgs() << "dim " << i;
      if (dimIsStructured(i))
        llvm::dbgs() << " structured\n";
      else
        llvm::dbgs() << " not strucuted\n";
    }
  }

  llvm::dbgs() << "\n";
}

LogicalResult PtrState::mulState(const PtrState &lhsState,
                                 const PtrState &rhsState,
                                 bool isAnalysisingUnstructured, Operation *op,
                                 OpBuilder &builder) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());

  auto loc = op->getLoc();

  
  
  if (lhsState.source && rhsState.source) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: do not support multiplying base pointers"));
    return failure();
  }

  
  if (!lhsState.scalar && !rhsState.scalar) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: only support multiplying pointer states when one of "
        "them represent a scalar"));
    return failure();
  }

  PtrState const *lhs = &lhsState;
  PtrState const *rhs = &rhsState;

  if (!rhs->scalar && lhs->scalar) {
    std::swap(lhs, rhs);
  }

  if (lhsState.scalar && rhsState.scalar) {
    scalar =
        builder.create<arith::MulIOp>(loc, lhsState.scalar, rhsState.scalar);
  }

  auto indexTy = IndexType::get(op->getContext());
  auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
  for (uint64_t i = 0; i < lhs->sizes.size(); i++) {
    if (lhs->dimIsStructured(i)) {
      OpFoldResult newOffset =
          mulOFRs(lhs->offsets[i], rhs->scalar, loc, builder);
      offsets.push_back(newOffset);
      OpFoldResult newStride =
          mulOFRs(lhs->strides[i], rhs->scalar, loc, builder);
      strides.push_back(newStride);
      OpFoldResult newShape = mulOFRs(lhs->shape[i], rhs->scalar, loc, builder);
      shape.push_back(newShape);
    } else {
      assert(!lhs->dimHasModulo(i) &&
             "should not have non-structured dimension with modulo");
      if (isAnalysisingUnstructured) {
        assert(!lhs->hasModulo() &&
               "should not have non-structured dimension with modulo");
        
        
        
        offsets.push_back(lhs->offsets[i]);
        
        OpFoldResult newStride =
            mulOFRs(lhs->strides[i], rhs->scalar, loc, builder);
        strides.push_back(newStride);
      } else {
        
        
        
        OpFoldResult newOffset = lhs->offsets[i];
        offsets.push_back(newOffset);
        
        OpFoldResult newStride = lhs->strides[i];
        strides.push_back(newStride);
      }
      
      shape.push_back(index0);
    }
    sizes.push_back(lhs->sizes[i]);
  }

  if (rhs->hasModulo()) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: do not support multiplying pointer states that has "
        "modulos"));
    return failure();
  }

  return success();
}

LogicalResult PtrState::mergeUnstructuredState(const PtrState &other,
                                               Operation *op) {
  if (isStructured() || other.isStructured()) {
    LLVM_DEBUG(op->emitRemark("Expect merging pointer states both of which are "
                              "unstructured, but got structured state"));
    return failure();
  }
  if (other.getNonStructuredDim() != getNonStructuredDim()) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: do not support merging pointer states with "
        "different non-structured dimensions"));
    return failure();
  }
  if (getRank() != other.getRank()) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: do not support merging pointer states with "
        "different ranks"));
    return failure();
  }
  int gatherDim = other.getNonStructuredDim();

  
  offsets[gatherDim] = other.offsets[gatherDim];
  strides[gatherDim] = other.strides[gatherDim];
  shape[gatherDim] = other.shape[gatherDim];

  return success();
}

tts::MakeTensorPtrOp PtrState::createTTSMakeTensorPtrOp(OpBuilder &builder,
                                                        Location loc) {
  SmallVector<int64_t> staticSizes;
  for (size_t i = 0; i < getRank(); i++) {
    auto s = getIntAttr(sizes[i]);
    assert(s.has_value());
    staticSizes.push_back(s.value());
  }

  auto op = builder.create<mlir::tts::MakeTensorPtrOp>(
      loc, source, staticSizes, strides, offsets, shape, order);
  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::make_tensor_ptr:\n";
    op->dump();
  });

  return op;
}

tts::MakeGatherScatterTensorPtrOp
PtrState::createTTSMakeGatherScatterTensorPtrOp(OpBuilder &builder,
                                                Location loc) {
  SmallVector<int64_t> staticSizes;
  for (size_t i = 0; i < getRank(); i++) {
    auto s = getIntAttr(sizes[i]);
    assert(s.has_value());
    staticSizes.push_back(s.value());
  }

  int nonContinuousDim = getNonStructuredDim();

  Value nonContinuousOffset = cast<Value>(offsets[nonContinuousDim]);

  
  auto offsetTy = cast<ShapedType>(nonContinuousOffset.getType());
  if (offsetTy.getRank() > 1) {
    SmallVector<ReassociationExprs, 4> reassociationMap(1);
    for (int i = 0; i < offsetTy.getRank(); ++i)
      reassociationMap[0].push_back(builder.getAffineDimExpr(i));

    int offsetSize = 1;
    for (int size : offsetTy.getShape())
      offsetSize *= size;

    auto collapseTy =
        RankedTensorType::get({offsetSize}, offsetTy.getElementType());
    nonContinuousOffset =
        builder
            .create<tensor::CollapseShapeOp>(
                loc, collapseTy, nonContinuousOffset, reassociationMap)
            .getResult();
    offsets[nonContinuousDim] = nonContinuousOffset;
  }
  
  auto op = builder.create<mlir::tts::MakeGatherScatterTensorPtrOp>(
      loc, source, nonContinuousOffset, nonContinuousDim, staticSizes, strides,
      offsets);
  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::make_gather_scatter_tensor_ptr:\n";
    op->dump();
  });

  return op;
}

LogicalResult PtrAnalysis::visitOperandAdd(arith::AddIOp addOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(addOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(addOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  
  if ((lhsState.getRank() == 1 && lhsState.hasModulo()) ||
      (rhsState.getRank() == 1 && rhsState.hasModulo())) {
    LLVM_DEBUG(addOp->emitRemark(
        "PtrAnalysis: do not support this pattern: a + arange(0, K) % M"));
    return failure();
  }

  
  
  if (!lhsState.isStructured() && rhsState.hasModulo()) {
    
    if (!enableMakeGatherScatterTensorPtr ||
        rhsState
            .rebuildAsGatherScatter(addOp.getRhs(),
                                    lhsState.getNonStructuredDim())
            .failed())
      return failure();
  } else if (lhsState.hasModulo() && !rhsState.isStructured()) {
    if (!enableMakeGatherScatterTensorPtr ||
        lhsState
            .rebuildAsGatherScatter(addOp.getLhs(),
                                    rhsState.getNonStructuredDim())
            .failed())
      return failure();
  }
  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "isAnalysisingUnstructured should not be true when "
           "enableMakeGatherScatterTensorPtr is false");
  }
  return state.addState(lhsState, rhsState, isAnalysisingUnstructured, addOp,
                        builder);
}

LogicalResult PtrAnalysis::visitOperandMul(arith::MulIOp mulOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(mulOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(mulOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  
  
  if (!lhsState.isStructured() && rhsState.hasModulo()) {
    
    if (!enableMakeGatherScatterTensorPtr ||
        rhsState
            .rebuildAsGatherScatter(mulOp.getRhs(),
                                    lhsState.getNonStructuredDim())
            .failed())
      return failure();
  } else if (lhsState.hasModulo() && !rhsState.isStructured()) {
    if (!enableMakeGatherScatterTensorPtr ||
        lhsState
            .rebuildAsGatherScatter(mulOp.getLhs(),
                                    rhsState.getNonStructuredDim())
            .failed())
      return failure();
  }

  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "isAnalysisingUnstructured should not be true when "
           "enableMakeGatherScatterTensorPtr is false");
  }
  return state.mulState(lhsState, rhsState, isAnalysisingUnstructured, mulOp,
                        builder);
}

LogicalResult PtrAnalysis::visitOperandRem(arith::RemSIOp remOp,
                                           PtrState &state, const Location loc,
                                           OpBuilder &builder) {
  assert(state.isEmpty());
  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "PtrAnalysis: isAnalysisingUnstructured should only be true "
           "when enableMakeGatherScatterTensorPtr is true");
    
    return state.rebuildAsUnsupportedOp(remOp.getResult());
  }

  PtrState rhsState;
  if (visitOperand(remOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  if (!rhsState.scalar) {
    LLVM_DEBUG(remOp->emitRemark(
        "PtrAnalysis: only support cases when rhs of remainder "
        "contains scalar"));
    return failure();
  }

  if (visitOperand(remOp.getLhs(), state, loc, builder).failed()) {
    return failure();
  }

  
  if (!state.isStructured()) {
    return state.rebuildAsGatherScatter(remOp.getResult(),
                                        state.getNonStructuredDim());
  }

  
  
  
  if (state.hasModulo()) {
    LLVM_DEBUG(remOp->emitRemark(
        "PtrAnalysis: do not support multiple modulo within an expression"));
    
    
    
    if (state.getRank() == 1 && enableMakeGatherScatterTensorPtr)
      
      
      return state.rebuildAsGatherScatter(remOp.getResult(), 0);
    else
      return failure();
  }

  if (state.getRank() == 1) {
    
    
    
    
    state.shape.back() = rhsState.scalar;
  } else if (state.getRank() == 2) {
    
    
    
    
    
    
    
    auto shape = cast<TensorType>(remOp.getResult().getType()).getShape();
    if (shape[0] == 1) {
      state.shape[1] = rhsState.scalar;
    } else if (shape[1] == 1) {
      state.shape[0] = rhsState.scalar;
    } else {
      LLVM_DEBUG(remOp->emitRemark(
          "PtrAnalysis: taking modulo on a 2D tensor with no singleton "
          "dimension not supported"));
      return failure();
    }
  } else {
    LLVM_DEBUG(remOp->emitRemark("PtrAnalysis: unsupported modulo pattern"));
    return failure();
  }
  return success();
}

LogicalResult PtrAnalysis::visitOperandExtSI(arith::ExtSIOp extOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  assert(state.isEmpty());
  return visitOperand(extOp.getIn(), state, loc, builder);
}

LogicalResult PtrAnalysis::visitOperandMakeRange(triton::MakeRangeOp rangeOp,
                                                 PtrState &state, Location loc,
                                                 OpBuilder &builder) {
  assert(state.isEmpty());

  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();

  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];
  assert(stride == 1 &&
         "Expect make_range op to always return tensor of stride 1");

  state.offsets.push_back(builder.getIndexAttr(start));
  state.sizes.push_back(builder.getIndexAttr(shape[0]));
  state.strides.push_back(builder.getIndexAttr(stride));
  state.shape.push_back(builder.getIndexAttr(0));
  return success();
}

LogicalResult
PtrAnalysis::visitOperandExpandDims(triton::ExpandDimsOp expandDimsOp,
                                    PtrState &state, const Location loc,
                                    OpBuilder &builder) {
  assert(state.isEmpty());

  if (visitOperand(expandDimsOp.getSrc(), state, loc, builder).failed()) {
    return failure();
  }

  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();

  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");

  
  state.offsets.insert(state.offsets.begin() + axis, builder.getIndexAttr(0));
  state.sizes.insert(state.sizes.begin() + axis, builder.getIndexAttr(1));
  state.strides.insert(state.strides.begin() + axis, builder.getIndexAttr(0));
  state.shape.insert(state.shape.begin() + axis, builder.getIndexAttr(0));

  if (state.hasModulo() && state.getRank() > 2) {
    LLVM_DEBUG(expandDimsOp->emitRemark(
        "PtrAnalysis: unsupported scenario where expand_dims result "
        "has modulo and rank > 2"));
    return failure();
  }

  return success();
}

LogicalResult
PtrAnalysis::visitOperandBroadcast(triton::BroadcastOp broadcastOp,
                                   PtrState &state, const Location loc,
                                   OpBuilder &builder) {
  assert(state.isEmpty());

  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();

  if (!isa<ShapedType>(src.getType())) {
    LLVM_DEBUG(broadcastOp->emitRemark(
        "PtrAnalysis: Unsupported broadcast source type"));
    return failure();
  }

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  if (visitOperand(src, state, loc, builder).failed()) {
    return failure();
  }

  for (size_t i = 0; i < dstShape.size(); i++) {
    if (srcShape[i] == dstShape[i]) {
      continue;
    } else if (srcShape[i] < dstShape[i]) {
      state.sizes[i] = builder.getIndexAttr(dstShape[i]);
    } else {
      llvm_unreachable("unexpected dimensions used in broadcast");
    }
  }
  return success();
}

LogicalResult PtrAnalysis::visitOperandSplat(triton::SplatOp splatOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  assert(state.isEmpty());

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  if (visitOperand(src, state, loc, builder).failed()) {
    return failure();
  }

  if (isa<IntegerType, IndexType, triton::PointerType>(src.getType())) {
    for (auto s : dstShape) {
      state.offsets.push_back(builder.getIndexAttr(0));
      state.sizes.push_back(builder.getIndexAttr(s));
      state.strides.push_back(builder.getIndexAttr(0));
      state.shape.push_back(builder.getIndexAttr(0));
    }
  } else {
    LLVM_DEBUG(splatOp->emitRemark("PtrAnalysis: unsupported splat pattern"));
    return failure();
  }

  
  
  
  if (state.scalar) {
    size_t scalarOffsetDim = 0;
    std::optional<size_t> singleNonSingletonDim;
    for (auto [idx, dim] : llvm::enumerate(dstShape)) {
      if (dim == 1)
        continue;
      if (singleNonSingletonDim) {
        singleNonSingletonDim = std::nullopt;
        break;
      }
      singleNonSingletonDim = idx;
    }
    if (singleNonSingletonDim)
      scalarOffsetDim = *singleNonSingletonDim;
    state.offsets[scalarOffsetDim] = state.scalar;
  }

  if (state.hasModulo() && state.getRank() > 2) {
    LLVM_DEBUG(splatOp->emitRemark(
        "PtrAnalysis: unsupported scenario where splat result "
        "has modulo and rank > 2"));
    return failure();
  }

  return success();
}

LogicalResult PtrAnalysis::visitOperandAddptr(triton::AddPtrOp addptrOp,
                                              PtrState &state,
                                              const Location loc,
                                              OpBuilder &builder) {
  assert(state.isEmpty());

  PtrState ptrState;
  if (visitOperand(addptrOp.getPtr(), ptrState, addptrOp.getLoc(), builder)
          .failed()) {
    return failure();
  } else if (!ptrState.source) {
    LLVM_DEBUG(llvm::dbgs()
               << "No src ptr state when processing " << addptrOp << "\n");
  }

  PtrState offsetState;
  if (visitOperand(addptrOp.getOffset(), offsetState, addptrOp.getLoc(),
                   builder)
          .failed()) {
    return failure();
  }

  assert(ptrState.source && "ptr field should provide source / base pointer");

  assert(ptrState.getRank() == offsetState.getRank() &&
         "ptr and offset field should have the same rank");

  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "isAnalysisingUnstructured should not be true when "
           "enableMakeGatherScatterTensorPtr is false");
  }
  return state.addState(ptrState, offsetState, isAnalysisingUnstructured,
                        addptrOp, builder);
}

LogicalResult PtrAnalysis::visitOperandConstSplat(arith::ConstantOp op,
                                                  PtrState &state,
                                                  const Location loc,
                                                  OpBuilder &builder) {
  assert(state.isEmpty());
  
  
  auto attr = cast<DenseElementsAttr>(op.getValue());
  auto elementType = attr.getElementType();
  assert(attr.isSplat() && isa<IntegerType>(elementType));
  auto values = attr.getValues<IntegerAttr>();
  auto value = values[0].getValue();
  auto constAttr = builder.getIndexAttr(value.getSExtValue());
  auto constOp = arith::ConstantOp::materialize(builder, constAttr,
                                                builder.getIndexType(), loc);

  state.scalar = constOp;

  auto resultType = cast<ShapedType>(op.getResult().getType());
  for (size_t i = 0; i < resultType.getShape().size(); i++) {
    if (i == 0) {
      state.offsets.push_back(constOp.getResult());
    } else {
      state.offsets.push_back(builder.getIndexAttr(0));
    }

    state.sizes.push_back(builder.getIndexAttr(resultType.getShape()[i]));
    state.strides.push_back(builder.getIndexAttr(0));
    state.shape.push_back(builder.getIndexAttr(0));
  }

  return success();
}

LogicalResult PtrAnalysis::visitOperandMakeTPtr(tts::MakeTensorPtrOp makeTPtrOp,
                                                PtrState &state,
                                                const Location loc,
                                                OpBuilder &builder) {

  assert(state.isEmpty());
  state.source = makeTPtrOp.getBase();
  state.offsets = makeTPtrOp.getMixedOffsets();
  state.sizes = makeTPtrOp.getMixedSizes();
  state.strides = makeTPtrOp.getMixedStrides();
  state.shape = makeTPtrOp.getMixedShape();
  state.order = SmallVector<int32_t>(makeTPtrOp.getOrder());

  return success();
}

LogicalResult
PtrAnalysis::visitOperandMakeTensorPtr(triton::MakeTensorPtrOp makeTPtrOp,
                                       PtrState &state, const Location loc,
                                       OpBuilder &builder) {
  assert(state.isEmpty());
  state.source = makeTPtrOp.getBase();

  if (makeTPtrOp.getOrder().empty()) {
    LLVM_DEBUG(makeTPtrOp->emitRemark(
        "PtrAnalysis: expect tt.make_tensor_ptr to have order field set"));
    return failure();
  }

  auto resType = cast<triton::PointerType>(makeTPtrOp.getResult().getType());
  auto pointeeType = cast<ShapedType>(resType.getPointeeType());
  auto shape = pointeeType.getShape();

  for (int64_t i = 0; i < pointeeType.getRank(); i++) {
    state.sizes.push_back(builder.getIndexAttr(shape[i]));

    auto strideCst = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), makeTPtrOp.getStrides()[i]);
    state.strides.push_back(strideCst.getResult());

    auto offsetCst = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), makeTPtrOp.getOffsets()[i]);

    auto scaledOffset = builder.create<arith::MulIOp>(
        loc, offsetCst.getResult(), strideCst.getResult());
    state.offsets.push_back(scaledOffset.getResult());

    auto shapeCst = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), makeTPtrOp.getShape()[i]);
    state.shape.push_back(shapeCst.getResult());
  }
  state.order = SmallVector<int32_t>(makeTPtrOp.getOrder());
  assert(state.isBlockPtr() &&
         "tt.make_tensor_ptr pointer state should describe a block pointer");

  return success();
}

LogicalResult PtrAnalysis::visitOperandForOp(scf::ForOp forOp, Value operand,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {

  auto it = llvm::find(forOp->getResults(), operand);
  auto index = std::distance(forOp->getResults().begin(), it);

  auto newState = getLoopResultPtrState(forOp, index);
  if (failed(newState)) {
    LLVM_DEBUG(forOp.emitWarning(
        "Rewrite for-op failed. Could not find PtrState returned by "
        "the loop."));
    return failure();
  }

  state = newState.value();
  return success();
}

LogicalResult PtrAnalysis::visitOperandIntToPtr(triton::IntToPtrOp op,
                                                PtrState &state,
                                                const Location loc,
                                                OpBuilder &builder) {
  state.source = op.getResult();
  return success();
}

LogicalResult PtrAnalysis::visitOperandBitcast(triton::BitcastOp op,
                                               PtrState &state,
                                               const Location loc,
                                               OpBuilder &builder) {
  auto resType = op.getResult().getType();
  if (isa<ShapedType>(resType)) {
    return visitOperand(op.getSrc(), state, loc, builder);
  }
  state.source = op.getResult();
  return success();
}

LogicalResult PtrAnalysis::visitOperand(Value operand, PtrState &state,
                                        const Location loc,
                                        OpBuilder &builder) {
  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "isAnalysisingUnstructured should not be true when "
           "enableMakeGatherScatterTensorPtr is false");
  }
  
  
  
  if (!isAnalysisingUnstructured &&
      knownPtrs.find(operand) != knownPtrs.end()) {
    state = knownPtrs.lookup(operand);
    return success();
  }

  if (isa<IntegerType>(operand.getType())) {
    OpBuilder::InsertionGuard guard(builder);
    if (!isa<BlockArgument>(operand) && operand.getDefiningOp()) {
      builder.setInsertionPointAfter(operand.getDefiningOp());
    }
    auto castOp = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), operand);
    state.scalar = castOp.getResult();
    return success();
  } else if (isa<IndexType>(operand.getType())) {
    state.scalar = operand;
    return success();
  }

  if (isa<triton::PointerType>(operand.getType())) {
    
    
    if (auto op = operand.getDefiningOp()) {
      if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
        return visitOperandAddptr(cast<triton::AddPtrOp>(op), state, loc,
                                  builder);
      } else if (auto castOp = dyn_cast<triton::BitcastOp>(op)) {
        return visitOperandBitcast(castOp, state, loc, builder);
      } else if (auto intToPtrOp = dyn_cast<triton::IntToPtrOp>(op)) {
        return visitOperandIntToPtr(intToPtrOp, state, loc, builder);
      } else if (auto makeTensorOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
        llvm_unreachable("Unexpected operand defining operation tts.make_tptr");
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        state.source = operand;
        return success();
      } else {
        LLVM_DEBUG(op->emitRemark(
            "Unexpected defining op for triton pointer operand"));
        return failure();
      }
    } else {
      state.source = operand;
      return success();
    }
  }

  if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    return visitOperandAdd(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::MulIOp>()) {
    return visitOperandMul(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    return visitOperandMakeRange(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return visitOperandBroadcast(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return visitOperandSplat(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return visitOperandExpandDims(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::AddPtrOp>()) {
    return visitOperandAddptr(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    return visitOperandConstSplat(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::RemSIOp>()) {
    return visitOperandRem(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ExtSIOp>()) {
    return visitOperandExtSI(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<scf::ForOp>()) {
    return visitOperandForOp(op, operand, state, loc, builder);
  } else if (!operand.getDefiningOp()) {
    if (!knownPtrs.contains(operand)) {
      return failure();
    }

    
    
    
    state = knownPtrs[operand];
    return success();
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "PtrAnalysis: encountered addptr operand produced by an "
                  "unsupported operation: "
               << operand);

    if (!enableMakeGatherScatterTensorPtr) {
      LLVM_DEBUG(llvm::dbgs()
                 << "PtrAnalysis: failed to rebuild as unsupported op\n");
      return failure();
    }
    return state.rebuildAsUnsupportedOp(operand);
  }
}

LogicalResult PtrAnalysis::rewriteAddptrOp(triton::AddPtrOp op) {
  OpBuilder builder(op);

  PtrState state;
  if (visitOperandAddptr(op, state, op.getLoc(), builder).failed()) {
    return failure();
  }

  knownPtrs[op.getResult()] = state;

  if (isa<RankedTensorType>(op.getPtr().getType())) {
    if (state.isStructured()) {
      auto maketptrOp = state.createTTSMakeTensorPtrOp(builder, op.getLoc());
      ptrMap.map(op.getResult(), maketptrOp.getResult());
    } else if (enableMakeGatherScatterTensorPtr) {
      PtrState unstructuredState;
      
      
      
      
      isAnalysisingUnstructured = true;
      
      
      LogicalResult result =
          visitOperandAddptr(op, unstructuredState, op.getLoc(), builder);
      
      isAnalysisingUnstructured = false;
      if (result.failed()) {
        LLVM_DEBUG(op->emitRemark(
            "PtrAnalysis: Failed to analyze ptr of tt.addptr for "
            "unstructured state"));
        return failure();
      }
      if (state.mergeUnstructuredState(unstructuredState, op).failed()) {
        LLVM_DEBUG(op->emitRemark(
            "PtrAnalysis: Failed to merge unstructured state for tt.addptr"));
        return failure();
      }
      auto maketptrOp =
          state.createTTSMakeGatherScatterTensorPtrOp(builder, op.getLoc());
      
      knownPtrs[op.getResult()] = state;
      ptrMap.map(op.getResult(), maketptrOp.getResult());
    } else {
      return failure();
    }
  } else {
    
    
    ptrMap.map(op.getResult(), op.getResult());
  }
  return success();
}

LogicalResult PtrAnalysis::rewriteMakeTensorPtrOp(triton::MakeTensorPtrOp op) {
  OpBuilder builder(op);

  PtrState state;
  if (visitOperandMakeTensorPtr(op, state, op.getLoc(), builder).failed()) {
    return failure();
  }

  auto maketptrOp = state.createTTSMakeTensorPtrOp(builder, op.getLoc());
  knownPtrs[op.getResult()] = state;
  ptrMap.map(op.getResult(), maketptrOp.getResult());
  return success();
}

LogicalResult PtrAnalysis::rewriteAdvanceOp(triton::AdvanceOp op) {
  OpBuilder builder(op);
  auto loc = op.getLoc();

  PtrState state;
  if (visitOperand(op->getOperand(0), state, loc, builder).failed()) {
    LLVM_DEBUG(
        op->emitRemark("PtrAnalysis: Failed to analyze ptr of tt.advance"));
    return failure();
  }
  assert(state.isBlockPtr() &&
         "tt.advance pointer state should describe a block pointer");

  auto incrementOffsets = op.getOffsets();

  SmallVector<OpFoldResult> newOffsets;
  for (auto [increment, offset, stride] :
       llvm::zip(incrementOffsets, state.offsets, state.strides)) {
    Value offsetValue;
    if (auto offsetIntAttr = getIntAttr(offset)) {
      auto constOp = builder.create<arith::ConstantOp>(
          loc, builder.getIndexAttr(offsetIntAttr.value()));
      offsetValue = constOp.getResult();
    } else {
      offsetValue = cast<Value>(offset);
    }
    auto castOp = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), increment);
    auto mulOp = builder.create<arith::MulIOp>(loc, castOp.getResult(),
                                               cast<Value>(stride));
    auto addOp =
        builder.create<arith::AddIOp>(loc, mulOp.getResult(), offsetValue);
    newOffsets.push_back(addOp.getResult());
  }

  state.offsets = SmallVector<OpFoldResult>(newOffsets);

  auto newOp = state.createTTSMakeTensorPtrOp(builder, loc);
  knownPtrs[op.getResult()] = state;
  ptrMap.map(op.getResult(), newOp.getResult());
  return success();
}

LogicalResult PtrAnalysis::rewriteBitcastOp(triton::BitcastOp op) {
  if (!isa<ShapedType>(op.getType()) ||
      !getTensorPtrPointeeElementType(op.getType())) {
    return success();
  }

  if (!isStorageCompatibleTensorPtrBitcast(op)) {
    return success();
  }

  auto mappedPtr = ptrMap.lookupOrNull(op.getSrc());
  if (!mappedPtr) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: source of storage-compatible bitcast is not mapped"));
    return failure();
  }

  auto stateIt = knownPtrs.find(op.getSrc());
  if (stateIt == knownPtrs.end()) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: source of storage-compatible bitcast has no PtrState"));
    return failure();
  }

  if (!stateIt->second.source) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: source of storage-compatible bitcast has no base ptr"));
    return failure();
  }

  OpBuilder builder(op);
  auto newSource =
      createBitcastedScalarBasePointer(stateIt->second.source, op.getType(),
                                       op.getLoc(), builder);
  if (failed(newSource)) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: failed to materialize base bitcast for tensor ptr"));
    return failure();
  }

  PtrState newState = stateIt->second;
  newState.source = *newSource;

  if (isa<tts::MakeGatherScatterTensorPtrOp>(mappedPtr.getDefiningOp())) {
    auto makePtrOp =
        newState.createTTSMakeGatherScatterTensorPtrOp(builder, op.getLoc());
    ptrMap.map(op.getResult(), makePtrOp.getResult());
  } else if (isa<tts::MakeTensorPtrOp>(mappedPtr.getDefiningOp())) {
    auto makePtrOp = newState.createTTSMakeTensorPtrOp(builder, op.getLoc());
    ptrMap.map(op.getResult(), makePtrOp.getResult());
  } else {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: source of storage-compatible bitcast is not structured"));
    return failure();
  }

  knownPtrs[op.getResult()] = newState;
  return success();
}

static bool isPointerType(Type t) {
  if (auto tensor = llvm::dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensor.getElementType());
  }
  return isa<triton::PointerType>(t);
}

FailureOr<PtrState> PtrAnalysis::getLoopInitArgPtrState(scf::ForOp forOp,
                                                        size_t index) {
  auto ptr = forOp.getInitArgs()[index];

  
  
  
  
  
  
  if (auto getStateOp = ptr.getDefiningOp<tts::GetStructuredStateOp>()) {
    auto originalPtr = getStateOp->getOperand(0);
    if (knownPtrs.count(originalPtr)) {
      return knownPtrs[originalPtr];
    }
  }

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  if (auto forOp = ptr.getDefiningOp<scf::ForOp>()) {
    return getLoopResultPtrState(forOp, index);
  }

  
  
  
  
  
  
  
  
  
  
  if (knownPtrs.count(ptr)) {
    assert(!ptr.getDefiningOp() && "Expect the ptr to be an iterarg");
    return knownPtrs[ptr];
  }

  return failure();
}

PtrState PtrAnalysis::reconcileLoopPtrState(
    scf::ForOp forOp, size_t iterArgIndex, const PtrState &state,
    llvm::function_ref<Value(scf::ForOp op, size_t)> getReplacementVal) {
  PtrState newState = state;
  int cnt = iterArgIndex + 1;
  if (newState.getRank() == 0) {
    assert(newState.scalar);
    
    
    newState.scalar = getReplacementVal(forOp, cnt);
  } else {
    for (auto &offset : newState.offsets) {
      offset = getReplacementVal(forOp, cnt++);
    }

    for (auto &stride : newState.strides) {
      stride = getReplacementVal(forOp, cnt++);
    }
  }

  return newState;
}

FailureOr<PtrState> PtrAnalysis::getLoopIterArgPtrState(scf::ForOp forOp,
                                                        size_t index) {
  auto state = getLoopInitArgPtrState(forOp, index);
  if (failed(state)) {
    return failure();
  }

  if (!state->isStructured()) {
    
    return failure();
  }

  return reconcileLoopPtrState(
      forOp, index, state.value(),
      [](scf::ForOp op, size_t index) { return op.getRegionIterArg(index); });
}

FailureOr<PtrState> PtrAnalysis::getLoopResultPtrState(scf::ForOp forOp,
                                                       size_t index) {
  auto state = getLoopInitArgPtrState(forOp, index);
  if (failed(state)) {
    return failure();
  }

  if (!state->isStructured()) {
    
    return failure();
  }
  return reconcileLoopPtrState(
      forOp, index, state.value(),
      [](scf::ForOp op, size_t index) { return op->getResult(index); });
}

LogicalResult PtrAnalysis::rewriteForOp(scf::ForOp op) {
  for (auto [i, arg] : llvm::enumerate(op.getRegionIterArgs())) {
    if (!maybeStructuredArgs.contains(arg)) {
      continue;
    }

    auto state = getLoopIterArgPtrState(op, i);
    if (failed(state)) {
      
      
      
      
      LLVM_DEBUG(op->emitWarning(
          "Rewrite for-op failed. Could not find PtrState for iter-arg index " +
          std::to_string(i)));
      continue;
    }
    
    if (state->noStructuredDimExists())
      continue;

    
    knownPtrs[arg] = state.value();

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    if (isPointerType(arg.getType())) {
      if (state->getRank() != 0) {
        OpBuilder builder(op.getRegion());
        auto maketptrOp = state->createTTSMakeTensorPtrOp(builder, op.getLoc());
        ptrMap.map(arg, maketptrOp.getResult());
      }
    }
  }

  
  if (rewriteOp(op).failed()) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: update loop body failed when rewriting for op"));
    return failure();
  }

  return success();
}

LogicalResult
PtrAnalysis::rewriteGetStructuredStateOp(tts::GetStructuredStateOp op) {
  auto tritonValue = op->getOperand(0);

  
  
  
  if (!knownPtrs.contains(tritonValue)) {
    LLVM_DEBUG(op.emitRemark(
        "Rewrite GetStructuredStateOp failed. Could not find PtrState."));
    op.getResult(0).replaceAllUsesWith(tritonValue);
    return failure();
  }

  tts::PtrState state = knownPtrs[tritonValue];
  if (!state.isStructured()) {
    LLVM_DEBUG(op.emitRemark(
        "Rewrite GetStructuredStateOp failed. PtrState is not structured."));
    op.getResult(0).replaceAllUsesWith(tritonValue);
    return failure();
  }
  Value remappedValue =
      ptrMap.contains(tritonValue) ? ptrMap.lookup(tritonValue) : tritonValue;

  SmallVector<Value> replacements{remappedValue};
  OpBuilder builder(op);

  if (state.getRank() == 0) {
    
    
    if (state.scalar) {
      replacements.push_back(state.scalar);
    } else {
      
      
      assert(!tritonValue.getDefiningOp());
      replacements.push_back(builder.create<arith::ConstantOp>(
          op.getLoc(), builder.getIndexAttr(0)));
    }
  } else {
    for (auto [j, s] : llvm::enumerate(state.offsets)) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        replacements.push_back(constOp.getResult());
      } else {
        replacements.push_back(cast<Value>(s));
      }
    }

    for (auto [j, s] : llvm::enumerate(state.strides)) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        replacements.push_back(constOp.getResult());
      } else {
        replacements.push_back(cast<Value>(s));
      }
    }
  }

  op->replaceAllUsesWith(replacements);
  op->erase();
  return success();
}

LogicalResult PtrAnalysis::rewriteLoadOp(triton::LoadOp op,
                                         bool useUnsafeMask) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto mask = op.getMask();
  auto other = op.getOther();
  auto loc = op.getLoc();

  if (!ptr) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: pointer is not replace with tts.make_tptr so "
        "loadOp cannot be rewritten"));
    return failure();
  }

  auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
    LLVM_DEBUG(
        op->emitRemark("PtrAnalysis: scalar loadOp will not be rewritten"));
    return failure();
  }

  ArrayRef<OpFoldResult> dims;
  mlir::triton::MaskState mstate(useUnsafeMask);
  Value scalarOther;

  OpBuilder builder(op);
  
  
  if (mask) {
    if (mstate.parse(mask, loc, builder).failed()) {
      LLVM_DEBUG(op->emitRemark("MaskAnalysis failed"));
      return failure();
    }
    ptr = applyUnstructuredMask(op, ptr, mstate, loc, builder);
    if (!ptr) {
      return failure();
    }
    dims = mstate.dims;
  }

  if (other) {
    assert(mask && "other value used while no masks are specified");

    scalarOther = utils::getScalarValue(other, loc, builder);
    if (!scalarOther) {
      LLVM_DEBUG(op->emitRemark("other value used in masked load produced by "
                                "unsupported instruction"));
      return failure();
    }
  }

  auto loadOp = builder.create<tts::LoadOp>(loc, ptr, dims, scalarOther);

  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::load:\n";
    loadOp->dump();
  });

  op.replaceAllUsesWith(loadOp.getResult());
  op->erase();
  return success();
}

























void PtrAnalysis::initializeMaybeStructuredArgs(Operation *op) {
  std::queue<Value> q;
  DenseSet<Value> visited;

  op->walk([&q, &visited](tts::GetStructuredStateOp getStateOp) {
    Value value = getStateOp->getResult(0);
    visited.insert(value);
    q.push(value);
  });

  while (!q.empty()) {
    auto v = q.front();
    q.pop();
    for (auto user : v.getUsers()) {
      
      
      
      
      
      
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        for (auto [argIndex, arg] :
             llvm::zip(llvm::index_range(0, forOp.getInitArgs().size()),
                       forOp.getInitArgs())) {
          if (arg != v) {
            continue;
          }
          auto iterArg = forOp.getRegionIterArg(argIndex);
          auto tiedLoopRes = forOp.getTiedLoopResult(iterArg);
          SmallVector<Value> neighbors{iterArg, tiedLoopRes};
          for (auto neighbor : neighbors) {
            maybeStructuredArgs.insert(neighbor);
            if (!visited.contains(neighbor)) {
              visited.insert(neighbor);
              q.push(neighbor);
            }
          }
        }
      } else {
        for (auto res : user->getResults()) {
          if (res.getType() != v.getType()) {
            continue;
          }
          maybeStructuredArgs.insert(res);
          if (!visited.contains(res)) {
            visited.insert(res);
            q.push(res);
          }
        }
      }
    }
  }
}

LogicalResult PtrAnalysis::rewriteStoreOp(triton::StoreOp op,
                                          bool useUnsafeMask) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto val = op.getValue();
  auto mask = op.getMask();
  auto loc = op.getLoc();

  if (!ptr) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: pointer is not replace with tts.make_tptr so "
        "storeOp cannot be rewritten"));
    return failure();
  }

  auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
    LLVM_DEBUG(
        op->emitRemark("PtrAnalysis: scalar storeOp will not be rewritten"));
    return failure();
  }

  ArrayRef<OpFoldResult> dims;
  mlir::triton::MaskState mstate(useUnsafeMask);

  OpBuilder builder(op);

  
  
  if (mask) {
    if (mstate.parse(mask, loc, builder).failed()) {
      LLVM_DEBUG(op->emitRemark("MaskAnalysis failed"));
      return failure();
    }
    ptr = applyUnstructuredMask(op, ptr, mstate, loc, builder);
    if (!ptr) {
      return failure();
    }
    dims = mstate.dims;
  }

  auto storeOp = builder.create<tts::StoreOp>(loc, ptr, val, dims);

  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::store:\n";
    storeOp->dump();
  });

  op->erase();
  return success();
}

LogicalResult PtrAnalysis::rewriteOp(Operation *rootOp, bool useUnsafeMask) {
  LLVM_DEBUG({
    llvm::dbgs() << "rewriting rootOp\n";
    rootOp->dump();
  });

  rootOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == rootOp) {
      return WalkResult::advance();
    }
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<triton::AddPtrOp>([&](auto addptr) {
          if (rewriteAddptrOp(addptr).failed()) {
            LLVM_DEBUG(
                addptr->emitRemark("PtrAnalysis: Failed to rewrite AddPtrOp"));
          }
          return WalkResult::advance();
        })
        .Case<triton::MakeTensorPtrOp>([&](auto maketptr) {
          if (rewriteMakeTensorPtrOp(maketptr).failed()) {
            LLVM_DEBUG(maketptr->emitRemark(
                "PtrAnalysis: Failed to rewrite MakeTensorPtrOp"));
          }
          return WalkResult::advance();
        })
        .Case<triton::AdvanceOp>([&](auto advance) {
          if (rewriteAdvanceOp(advance).failed()) {
            LLVM_DEBUG(advance->emitRemark(
                "PtrAnalysis: Failed to rewrite AdvanceOp"));
          }
          return WalkResult::advance();
        })
        .Case<triton::BitcastOp>([&](auto bitcast) {
          if (rewriteBitcastOp(bitcast).failed()) {
            LLVM_DEBUG(bitcast->emitRemark(
                "PtrAnalysis: Failed to rewrite BitcastOp"));
          }
          return WalkResult::advance();
        })
        .Case<triton::LoadOp>([&](auto load) {
          if (rewriteLoadOp(load, useUnsafeMask).failed()) {
            LLVM_DEBUG(
                load->emitRemark("PtrAnalysis: Failed to rewrite LoadOp"));
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<triton::StoreOp>([&](auto store) {
          if (rewriteStoreOp(store, useUnsafeMask).failed()) {
            LLVM_DEBUG(
                store->emitRemark("PtrAnalysis: Failed to rewrite StoreOp"));
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<scf::ForOp>([&](auto forOp) {
          
          
          
          
          if (rewriteForOp(forOp).failed()) {
            LLVM_DEBUG(
                forOp->emitRemark("PtrAnalysis: Failed to rewrite ForOp"));
          }
          return WalkResult::skip();
        })
        .Case<tts::GetStructuredStateOp>(
            [&](tts::GetStructuredStateOp getStateOp) {
              
              
              
              
              
              
              
              
              
              
              auto tritonValue = getStateOp->getOperand(0);
              if (!knownPtrs.contains(tritonValue)) {
                PtrState state;
                OpBuilder b(getStateOp);
                if (succeeded(visitOperand(tritonValue, state,
                                           getStateOp->getLoc(), b)) &&
                    state.isStructured()) {
                  knownPtrs[tritonValue] = state;
                } else {
                  LLVM_DEBUG(getStateOp->emitRemark(
                      "PtrAnalysis: Failed to populate ptr "
                      "state for tensor of indices"));
                }
              }

              return WalkResult::skip();
            })
        .Default([&](auto) { return WalkResult::advance(); });
  });

  return success();
}

} 
} 

//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/PerfMatmul/PerfMatmul.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"

#define DEBUG_TYPE "perf-matmul"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PERFMATMUL
#include "triton-shared/Conversion/PerfMatmul/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class PerfMatmulPass : public triton::impl::PerfMatmulBase<PerfMatmulPass> {
  static constexpr unsigned kProgramInfoArgCount = 6;
  static constexpr unsigned kProgramIdArgCount = 3;
  static constexpr llvm::StringLiteral kReadCycleFuncName = "read_cycle";

  static Value createReadCycleCall(OpBuilder &builder, Location loc) {
    auto call = LLVM::CallOp::create(
        builder, loc, TypeRange{builder.getI64Type()},
        SymbolRefAttr::get(builder.getContext(), kReadCycleFuncName),
        ValueRange{});
    return call->getResult(0);
  }

  static FailureOr<Value> castToIndex(OpBuilder &builder, Location loc,
                                      Value value) {
    if (value.getType().isIndex()) {
      return value;
    }
    if (isa<IntegerType>(value.getType())) {
      return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                                 value)
          .getResult();
    }
    return failure();
  }

  static bool
  isKernelFunction(func::FuncOp func,
                   const llvm::SmallDenseSet<llvm::StringRef> &called) {
    auto funcName = func.getSymNameAttr().getValue();
    if (func.isDeclaration()) {
      return false;
    }
    if (funcName == kReadCycleFuncName) {
      return false;
    }
    if (func.isPrivate()) {
      return false;
    }
    if (called.contains(funcName)) {
      return false;
    }
    return true;
  }

  static LLVM::LLVMFuncOp getOrCreateReadCycleFunc(ModuleOp module) {
    if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(kReadCycleFuncName)) {
      return func;
    }

    OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());

    auto funcType = LLVM::LLVMFunctionType::get(
        builder.getI64Type(), ArrayRef<Type>{}, /*isVarArg=*/false);
    auto readCycle =
        LLVM::LLVMFuncOp::create(builder, module.getLoc(), kReadCycleFuncName,
                                 funcType);

    Block *entry = readCycle.addEntryBlock(builder);
    OpBuilder bodyBuilder(entry, entry->begin());

    auto asmDialectAttr = LLVM::AsmDialectAttr::get(
        module.getContext(), LLVM::AsmDialect::AD_ATT);
    auto cycle = LLVM::InlineAsmOp::create(
        bodyBuilder, module.getLoc(), TypeRange{bodyBuilder.getI64Type()},
        ValueRange{},
        /*asm_string=*/"rdcycle $0", /*constraints=*/"=r",
        /*has_side_effects=*/true, /*is_align_stack=*/false,
        LLVM::TailCallKind::None, asmDialectAttr, ArrayAttr());

    Value cycleValue = cycle->getResult(0);
    LLVM::ReturnOp::create(bodyBuilder, module.getLoc(),
                           ValueRange{cycleValue});
    return readCycle;
  }

  FailureOr<unsigned> addMatmulCounterArgument(func::FuncOp func) const {
    auto oldType = func.getFunctionType();
    auto oldInputs = oldType.getInputs();

    if (oldInputs.size() < kProgramInfoArgCount) {
      func.emitError()
          << "expected at least " << kProgramInfoArgCount
          << " trailing launch grid/program id args, but got "
          << oldInputs.size();
      return failure();
    }

    unsigned insertIdx = oldInputs.size() - kProgramInfoArgCount;
    auto counterType =
        MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic,
                         ShapedType::kDynamic},
                        IntegerType::get(func.getContext(), 64));

    SmallVector<Type> newInputs(oldInputs);
    newInputs.insert(newInputs.begin() + insertIdx, counterType);
    auto newType = FunctionType::get(func.getContext(), newInputs,
                                     oldType.getResults());
    func.setFunctionType(newType);

    if (func.getAllArgAttrs()) {
      SmallVector<DictionaryAttr> newArgAttrs;
      func.getAllArgAttrs(newArgAttrs);
      newArgAttrs.insert(newArgAttrs.begin() + insertIdx, DictionaryAttr());
      func.setAllArgAttrs(newArgAttrs);
    }

    func.getBody().front().insertArgument(insertIdx, counterType,
                                          func.getLoc());

    return insertIdx;
  }

  LogicalResult instrumentKernel(func::FuncOp func, unsigned counterArgIdx) {
    if (func.getNumArguments() < kProgramIdArgCount) {
      func.emitError() << "expected at least " << kProgramIdArgCount
                       << " program id args";
      return failure();
    }

    Value counterArg = func.getArgument(counterArgIdx);
    auto counterType = dyn_cast<MemRefType>(counterArg.getType());
    if (!counterType || counterType.getRank() != 3 ||
        counterType.getElementType() != IntegerType::get(func.getContext(), 64)) {
      func.emitError() << "matmul counter argument must be memref<?x?x?xi64>";
      return failure();
    }

    Block &entry = func.getBody().front();
    OpBuilder entryBuilder(func.getContext());
    entryBuilder.setInsertionPointToStart(&entry);

    auto i64Type = entryBuilder.getI64Type();
    auto accType = MemRefType::get({}, i64Type);
    auto acc = entryBuilder.create<memref::AllocaOp>(func.getLoc(), accType);
    auto zero = entryBuilder.create<arith::ConstantIntOp>(func.getLoc(), 0, 64);
    entryBuilder.create<memref::StoreOp>(func.getLoc(), zero, acc,
                                         ValueRange{});

    SmallVector<linalg::MatmulOp> matmuls;
    func.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });

    for (auto matmul : matmuls) {
      OpBuilder before(matmul);
      Value t0 = createReadCycleCall(before, matmul.getLoc());

      OpBuilder after(matmul);
      after.setInsertionPointAfter(matmul);
      Value t1 = createReadCycleCall(after, matmul.getLoc());
      Value dt = after.create<arith::SubIOp>(matmul.getLoc(), t1, t0);
      Value prev = after.create<memref::LoadOp>(matmul.getLoc(), acc,
                                                ValueRange{});
      Value updated = after.create<arith::AddIOp>(matmul.getLoc(), prev, dt);
      after.create<memref::StoreOp>(matmul.getLoc(), updated, acc,
                                    ValueRange{});
    }

    Value xId = func.getArgument(func.getNumArguments() - 3);
    Value yId = func.getArgument(func.getNumArguments() - 2);
    Value zId = func.getArgument(func.getNumArguments() - 1);

    SmallVector<func::ReturnOp> returns;
    func.walk([&](func::ReturnOp ret) { returns.push_back(ret); });

    for (auto ret : returns) {
      OpBuilder builder(ret);
      auto xIdIdx = castToIndex(builder, ret.getLoc(), xId);
      auto yIdIdx = castToIndex(builder, ret.getLoc(), yId);
      auto zIdIdx = castToIndex(builder, ret.getLoc(), zId);
      if (failed(xIdIdx) || failed(yIdIdx) || failed(zIdIdx)) {
        ret.emitError() << "expected pid args to be integer/index types";
        return failure();
      }

      Value total = builder.create<memref::LoadOp>(ret.getLoc(), acc,
                                                   ValueRange{});
      builder.create<memref::StoreOp>(ret.getLoc(), total, counterArg,
                                      ValueRange{*xIdIdx, *yIdIdx, *zIdIdx});
    }

    return success();
  }

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    llvm::SmallDenseSet<llvm::StringRef> calledFuncs;
    module.walk([&](func::CallOp callOp) {
      calledFuncs.insert(callOp.getCalleeAttr().getValue());
    });

    SmallVector<func::FuncOp> kernels;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (isKernelFunction(func, calledFuncs)) {
        kernels.push_back(func);
      }
    }

    if (kernels.empty()) {
      return;
    }

    getOrCreateReadCycleFunc(module);

    llvm::DenseMap<Operation *, unsigned> counterArgIndices;
    for (auto func : kernels) {
      auto counterArgIdx = addMatmulCounterArgument(func);
      if (failed(counterArgIdx)) {
        signalPassFailure();
        return;
      }
      counterArgIndices.insert({func.getOperation(), *counterArgIdx});
    }

    for (auto func : kernels) {
      if (failed(instrumentKernel(func, counterArgIndices[func.getOperation()]))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createPerfMatmulPass() {
  return std::make_unique<PerfMatmulPass>();
}

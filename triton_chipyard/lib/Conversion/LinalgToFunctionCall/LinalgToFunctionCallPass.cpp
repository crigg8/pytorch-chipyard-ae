#include "triton-shared/Conversion/LinalgToFunctionCall/Passes.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <utility>

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_LINALGTOFUNCTIONCALL
#include "triton-shared/Conversion/LinalgToFunctionCall/Passes.h.inc"
} // namespace mlir::triton

namespace {

struct OperandTypeConstraint {
  Type elementType;
  SmallVector<int64_t> physicalRanks;
  int64_t logicalRank = 0;
  bool squeezeStaticUnitDims = false;
  bool canonicalizeToLogicalRank = false;
};

static FailureOr<SmallVector<OperandTypeConstraint>>
parseOperandTypeConstraints(StringRef encoded, MLIRContext *context) {
  SmallVector<OperandTypeConstraint> constraints;
  if (encoded.empty())
    return constraints;

  SmallVector<StringRef> encodedOperands;
  encoded.split(encodedOperands, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef encodedOperand : encodedOperands) {
    SmallVector<StringRef> fields;
    encodedOperand.split(fields, ':', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
    if (fields.size() != 5)
      return failure();

    Type elementType = parseType(fields[0], context);
    if (!elementType)
      return failure();

    SmallVector<StringRef> encodedRanks;
    fields[1].split(encodedRanks, '|', /*MaxSplit=*/-1,
                    /*KeepEmpty=*/false);
    if (encodedRanks.empty())
      return failure();
    SmallVector<int64_t> physicalRanks;
    for (StringRef encodedRank : encodedRanks) {
      int64_t rank = -1;
      if (encodedRank.getAsInteger(10, rank) || rank < 0)
        return failure();
      physicalRanks.push_back(rank);
    }

    int64_t logicalRank = -1;
    if (fields[2].getAsInteger(10, logicalRank) || logicalRank < 0)
      return failure();
    bool squeezeStaticUnitDims;
    if (fields[3] == "preserve")
      squeezeStaticUnitDims = false;
    else if (fields[3] == "squeeze_static")
      squeezeStaticUnitDims = true;
    else
      return failure();
    bool canonicalizeToLogicalRank;
    if (fields[4] == "0")
      canonicalizeToLogicalRank = false;
    else if (fields[4] == "1")
      canonicalizeToLogicalRank = true;
    else
      return failure();
    if (canonicalizeToLogicalRank &&
        (!squeezeStaticUnitDims || logicalRank == 0))
      return failure();

    constraints.push_back(OperandTypeConstraint{
        elementType, std::move(physicalRanks), logicalRank,
        squeezeStaticUnitDims, canonicalizeToLogicalRank});
  }
  return constraints;
}

static bool hasPhysicalRank(const OperandTypeConstraint &constraint,
                            int64_t rank) {
  for (int64_t allowedRank : constraint.physicalRanks) {
    if (allowedRank == rank)
      return true;
  }
  return false;
}

static bool matchOperandType(MemRefType type,
                             const OperandTypeConstraint &constraint,
                             SmallVectorImpl<int64_t> &canonicalShape) {
  int64_t rank = type.getRank();
  if (!hasPhysicalRank(constraint, rank) ||
      type.getElementType() != constraint.elementType ||
      constraint.logicalRank > rank)
    return false;

  ArrayRef<int64_t> shape = type.getShape();
  canonicalShape.assign(shape.begin(), shape.end());
  if (!constraint.squeezeStaticUnitDims)
    return rank == constraint.logicalRank;

  int64_t dimsToDrop = rank - constraint.logicalRank;
  for (int64_t dim = 0;
       dim < static_cast<int64_t>(canonicalShape.size()) && dimsToDrop > 0;) {
    if (canonicalShape[dim] == 1) {
      canonicalShape.erase(canonicalShape.begin() + dim);
      --dimsToDrop;
      continue;
    }
    ++dim;
  }
  return dimsToDrop == 0;
}

class LinalgToFunctionCallPass
    : public mlir::triton::impl::LinalgToFunctionCallBase<
          LinalgToFunctionCallPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (targetOp.empty() || !llvm::StringRef(targetOp).starts_with("linalg.")) {
      module.emitError() << "target-op must be an exact linalg.* operation name";
      signalPassFailure();
      return;
    }
    if (targetOp == "linalg.generic") {
      module.emitError()
          << "linalg.generic requires an indexing/body predicate and is not "
             "supported by linalg-to-function-call";
      signalPassFailure();
      return;
    }
    if (functionName.empty()) {
      module.emitError() << "function must be a non-empty external symbol";
      signalPassFailure();
      return;
    }
    FailureOr<SmallVector<OperandTypeConstraint>> parsedConstraints =
        parseOperandTypeConstraints(operandTypes, &getContext());
    if (failed(parsedConstraints)) {
      module.emitError()
          << "invalid operand-types; expected "
             "element:ranks:logical-rank:unit-policy:canonicalize entries";
      signalPassFailure();
      return;
    }

    SmallVector<Operation *> matches;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == targetOp)
        matches.push_back(op);
    });

    for (Operation *op : matches) {
      if (!isa<linalg::LinalgOp>(op)) {
        op->emitError() << "target operation does not implement LinalgOp";
        signalPassFailure();
        return;
      }
      auto dps = dyn_cast<DestinationStyleOpInterface>(op);
      if (!dps) {
        op->emitError()
            << "target operation does not implement DestinationStyleOpInterface";
        signalPassFailure();
        return;
      }
      if (op->getNumResults() != 0) {
        op->emitError()
            << "target must be bufferized before linalg-to-function-call";
        signalPassFailure();
        return;
      }

      SmallVector<Value> orderedOperands;
      for (OpOperand *operand : dps.getDpsInputOperands())
        orderedOperands.push_back(operand->get());
      for (Value operand : dps.getDpsInits())
        orderedOperands.push_back(operand);

      SmallVector<SmallVector<int64_t>> canonicalShapes;
      if (!parsedConstraints->empty()) {
        if (parsedConstraints->size() != orderedOperands.size())
          continue;
        bool typesMatch = true;
        canonicalShapes.resize(orderedOperands.size());
        for (auto [index, operand] : llvm::enumerate(orderedOperands)) {
          auto memrefType = dyn_cast<MemRefType>(operand.getType());
          if (!memrefType ||
              !matchOperandType(memrefType, (*parsedConstraints)[index],
                                canonicalShapes[index])) {
            typesMatch = false;
            break;
          }
        }
        if (!typesMatch)
          continue;
      }

      OpBuilder builder(op);
      SmallVector<Value> callOperands;
      SmallVector<Type> functionInputs;
      bool canonicalizationFailed = false;
      for (auto [index, rawOperand] : llvm::enumerate(orderedOperands)) {
        Value operand = rawOperand;
        if (!parsedConstraints->empty() &&
            (*parsedConstraints)[index].canonicalizeToLogicalRank &&
            cast<MemRefType>(operand.getType()).getRank() !=
                (*parsedConstraints)[index].logicalRank) {
          FailureOr<Value> canonicalized = memref::SubViewOp::rankReduceIfNeeded(
              builder, op->getLoc(), operand, canonicalShapes[index]);
          if (failed(canonicalized)) {
            canonicalizationFailed = true;
            break;
          }
          operand = *canonicalized;
        }
        Type type = operand.getType();
        if (auto ranked = dyn_cast<MemRefType>(type)) {
          auto unranked = UnrankedMemRefType::get(
              ranked.getElementType(), ranked.getMemorySpace());
          callOperands.push_back(builder.create<memref::CastOp>(
              op->getLoc(), unranked, operand));
          functionInputs.push_back(unranked);
          continue;
        }
        if (isa<UnrankedMemRefType>(type)) {
          callOperands.push_back(operand);
          functionInputs.push_back(type);
          continue;
        }
        if (type.isIntOrIndexOrFloat()) {
          callOperands.push_back(operand);
          functionInputs.push_back(type);
          continue;
        }
        op->emitError() << "unsupported external-call operand type: " << type;
        signalPassFailure();
        return;
      }
      if (canonicalizationFailed) {
        op->emitError() << "failed to canonicalize matched memref rank";
        signalPassFailure();
        return;
      }

      auto expectedType =
          FunctionType::get(&getContext(), functionInputs, TypeRange{});
      Operation *existing = module.lookupSymbol(functionName);
      func::FuncOp declaration;
      if (existing) {
        declaration = dyn_cast<func::FuncOp>(existing);
        if (!declaration || !declaration.isDeclaration() ||
            declaration.getFunctionType() != expectedType) {
          op->emitError()
              << "external symbol " << functionName
              << " already exists with an incompatible definition or type";
          signalPassFailure();
          return;
        }
      } else {
        OpBuilder moduleBuilder(module.getContext());
        moduleBuilder.setInsertionPointToStart(module.getBody());
        declaration = func::FuncOp::create(moduleBuilder, op->getLoc(),
                                            functionName, expectedType);
        declaration.setPrivate();
      }

      builder.create<func::CallOp>(op->getLoc(), declaration,
                                   callOperands);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createLinalgToFunctionCallPass() {
  return std::make_unique<LinalgToFunctionCallPass>();
}

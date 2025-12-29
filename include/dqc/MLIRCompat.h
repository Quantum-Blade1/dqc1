// Compatibility utilities for MLIR API differences
#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

// No-op macro present in some MLIR versions
#ifndef MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_OPNAME_ALLOCATIONFN
#define MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_OPNAME_ALLOCATIONFN(x)
#endif

namespace dqc {
namespace compat {

inline mlir::Operation *createOperation(mlir::OpBuilder &builder,
                                        mlir::OperationState &state) {
  auto *op = mlir::Operation::create(state);
  builder.getBlock()->getOperations().insert(builder.getInsertionPoint(),
                                             op);
  return op;
}

inline mlir::Operation *createOperation(mlir::PatternRewriter &rewriter,
                                        mlir::OperationState &state) {
  return createOperation(static_cast<mlir::OpBuilder &>(rewriter), state);
}

} // namespace compat
} // namespace dqc

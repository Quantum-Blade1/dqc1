//===- DQCDialect.cpp - DQC Dialect Implementation ---*- C++ -*-===//
//
// This file implements the DQC dialect.
//
//===----------------------------------------------------------===//

#include "dqc/DQCDialect.h"
#include "dqc/DQCOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace dqc;

//===------------------------------------------------------===//
// DQC dialect
//===------------------------------------------------------===//

#include "dqc/DQCDialect.cpp.inc"

namespace dqc {

void DQCDialect::initialize() {
  // Register types
  addTypes<QubitType, EPRHandleType>();

  // Register operations
  addOperations<
#define GET_OP_LIST
#include "dqc/DQCOps.cpp.inc"
      >();
}

//===------------------------------------------------------===//
// Type Definitions
//===------------------------------------------------------===//

Type DQCDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return nullptr;

  if (keyword == "qubit") {
    return QubitType::get(getContext());
  } else if (keyword == "epr_handle") {
    return EPRHandleType::get(getContext());
  }

  parser.emitError(parser.getNameLoc()) << "unknown DQC type: " << keyword;
  return nullptr;
}

void DQCDialect::printType(Type type, DialectAsmPrinter &printer) const {
  mlir::TypeSwitch<Type>(type)
      .Case<QubitType>([&](Type) { printer << "qubit"; })
      .Case<EPRHandleType>([&](Type) { printer << "epr_handle"; })
      .Default([](Type) { llvm_unreachable("unknown DQC type"); });
}

} // namespace dqc

::mlir::Operation *dqc::DQCDialect::materializeConstant(::mlir::OpBuilder &builder,
                                                       ::mlir::Attribute value,
                                                       ::mlir::Type type,
                                                       ::mlir::Location loc) {
  // Simple materialization: forward to arith.constant when possible.
  if (auto intAttr = value.dyn_cast<::mlir::IntegerAttr>()) {
    return builder.create<::mlir::arith::ConstantOp>(loc, type, intAttr).getOperation();
  }
  if (auto floatAttr = value.dyn_cast<::mlir::FloatAttr>()) {
    return builder.create<::mlir::arith::ConstantOp>(loc, type, floatAttr).getOperation();
  }
  return nullptr;
}

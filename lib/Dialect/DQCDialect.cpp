//===- DQCDialect.cpp - DQC Dialect Implementation ---*- C++ -*-===//
//
// This file implements the DQC dialect.
//
//===----------------------------------------------------------===//

#include "dqc/DQCDialect.h"
#include "dqc/DQCOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace llvm;
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
  StringRef keyword;
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
  TypeSwitch<Type>(type)
      .Case<QubitType>([&](Type) { printer << "qubit"; })
      .Case<EPRHandleType>([&](Type) { printer << "epr_handle"; })
      .Default([](Type) { llvm_unreachable("unknown DQC type"); });
}

} // namespace dqc

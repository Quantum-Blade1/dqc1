//===- DQCDialect.h - DQC Dialect Declaration ---*- C++ -*-===//
//
// This file defines the DQC dialect.
//
//===------------------------------------------------------===//

#ifndef DQC_DQCDIALECT_H
#define DQC_DQCDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"

#if MLIR_MAJOR_VERSION >= 18
#include "mlir/Bytecode/BytecodeOpInterface.h"
#endif


namespace dqc {
// DQC types
class QubitType
    : public mlir::Type::TypeBase<QubitType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
};

class EPRHandleType : public mlir::Type::TypeBase<EPRHandleType, mlir::Type,
                                                  mlir::TypeStorage> {
public:
  using Base::Base;
};

} // namespace dqc

#define GET_DIALECT_CLASSES
#include "dqc/DQCDialect.h.inc"

#endif // DQC_DQCDIALECT_H

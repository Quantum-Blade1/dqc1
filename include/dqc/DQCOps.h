//===- DQCOps.h - DQC Op Declarations ---*- C++ -*-===//
//
// This file declares the operations in the DQC dialect.
//
//===------------------------------------------------===//

#ifndef DQC_DQCOPS_H
#define DQC_DQCOPS_H

#include "dqc/DQCDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "dqc/DQCOps.h.inc"

#endif // DQC_DQCOPS_H

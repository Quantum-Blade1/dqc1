//===- MPIDialect.h - MPI Dialect Declarations -----------*- C++ -*-===//
//
// Lowering target dialect for distributed quantum execution.
// Operations map to MPI communication patterns for EPR distribution
// and telegate sequencing.
//
//===--------------------------------------------------------------===//

#pragma once

#include "dqc/DQCDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "dqc/MPIDialect.h.inc"

#define GET_OP_CLASSES
#include "dqc/MPIOps.h.inc"

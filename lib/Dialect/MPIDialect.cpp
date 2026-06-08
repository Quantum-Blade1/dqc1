//===- MPIDialect.cpp - MPI Dialect Implementation ------*- C++ -*-===//
//===--------------------------------------------------------------===//

#include "dqc/MPIDialect.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace dqc::mpi;

#include "dqc/MPIDialect.cpp.inc"

void MPIDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dqc/MPIOps.cpp.inc"
  >();
}

#define GET_OP_CLASSES
#include "dqc/MPIOps.cpp.inc"

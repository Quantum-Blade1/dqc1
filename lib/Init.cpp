//===- Init.cpp - DQC Compiler Initialization ---*- C++ -*-===//
//
// This file initializes the DQC compiler library.
//
//===------------------------------------------------------===//

#include "dqc/DQCDialect.h"
#include "dqc/DQCOps.h"
#include "dqc/Passes.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;

namespace dqc {
/// Initialize the DQC compiler by registering dialects and passes
void initDQCCompiler(mlir::MLIRContext *context) {
  context->loadDialect<DQCDialect>();
  registerDQCPasses();
}
} // namespace dqc

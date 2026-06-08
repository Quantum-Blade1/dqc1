
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "dqc/DQCDialect.h"
#include "dqc/MPIDialect.h"
#include "dqc/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace llvm;

int main(int argc, char **argv) {
  // Register DQC passes
  dqc::registerDQCPasses();

  mlir::DialectRegistry registry;
  registry.insert<dqc::DQCDialect>();
  registry.insert<dqc::mpi::MPIDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<LLVM::LLVMDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "DQC modular optimizer driver\n", registry));
}

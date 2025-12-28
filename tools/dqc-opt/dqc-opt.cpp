
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
#include "dqc/Passes.h"

using namespace mlir;
using namespace llvm;

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  // Register DQC passes
  mlir::dqc::registerDQCPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::dqc::DQCDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "DQC modular optimizer driver\n", registry));
}

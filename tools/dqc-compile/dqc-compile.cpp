//===- dqc-compile.cpp - DQC Full Compilation Driver ----*- C++ -*-===//
//
// Compiles a DQC MLIR file to LLVM IR by running the full five-pass
// pipeline and translating the resulting LLVM dialect to textual IR.
// The output .ll can be linked with clang against libdqc_runtime.
//
//===--------------------------------------------------------------===//

#include "dqc/DQCDialect.h"
#include "dqc/MPIDialect.h"
#include "dqc/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;

static cl::opt<std::string> inputFilename(cl::Positional,
                                           cl::desc("<input .mlir file>"),
                                           cl::Required);

static cl::opt<std::string> outputFilename("o",
                                            cl::desc("Output file (.ll)"),
                                            cl::value_desc("filename"),
                                            cl::init("-"));

static cl::opt<bool> quiet("q",
                            cl::desc("Suppress informational messages"),
                            cl::init(false));

static cl::opt<bool> verifyOption("verify",
                                   cl::desc("Verify circuit equivalence"),
                                   cl::init(false));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "DQC Compiler\n");

  MLIRContext context;
  context.loadDialect<dqc::DQCDialect>();
  context.loadDialect<dqc::mpi::MPIDialect>();
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<arith::ArithDialect>();
  context.loadDialect<LLVM::LLVMDialect>();

  // Parse input.
  auto sourceMgr = std::make_shared<SourceMgr>();
  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (auto ec = fileOrErr.getError()) {
    errs() << "Error opening '" << inputFilename << "': "
           << ec.message() << "\n";
    return 1;
  }
  sourceMgr->AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(*sourceMgr, &context);
  if (!module) {
    errs() << "Error parsing input file\n";
    return 1;
  }

  // Run the full pass pipeline.
  PassManager pm(&context);
  pm.addNestedPass<func::FuncOp>(dqc::createInteractionGraphPass());
  pm.addNestedPass<func::FuncOp>(dqc::createCCXDecompositionPass());
  pm.addNestedPass<func::FuncOp>(dqc::createTeleGateSynthesisPass());
  pm.addNestedPass<func::FuncOp>(dqc::createGreedyReorderingPass(verifyOption));
  pm.addPass(dqc::createMPILoweringPass());
  pm.addPass(dqc::createLLVMLoweringPass());

  if (failed(pm.run(*module))) {
    errs() << "Pass pipeline failed\n";
    return 1;
  }

  // Translate to LLVM IR.
  registerBuiltinDialectTranslation(context);
  registerLLVMDialectTranslation(context);

  LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    errs() << "Failed to translate to LLVM IR\n";
    return 1;
  }

  llvmModule->setModuleIdentifier("dqc_circuit");
  llvmModule->setSourceFileName(inputFilename);

  // Generate a main() wrapper if the circuit isn't already named main.
  if (!llvmModule->getFunction("main")) {
    llvm::Function *entry = nullptr;
    for (auto &F : *llvmModule)
      if (!F.isDeclaration() && F.getName() != "main") {
        entry = &F;
        break;
      }

    if (entry) {
      auto *mainTy = llvm::FunctionType::get(
          llvm::Type::getInt32Ty(llvmContext), false);
      auto *mainFn = llvm::Function::Create(
          mainTy, llvm::GlobalValue::ExternalLinkage, "main", *llvmModule);
      auto *bb = llvm::BasicBlock::Create(llvmContext, "entry", mainFn);
      llvm::IRBuilder<> builder(bb);
      builder.CreateCall(entry);
      builder.CreateRet(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(llvmContext), 0));
      if (!quiet)
        errs() << "Generated main() wrapper calling @"
               << entry->getName() << "\n";
    }
  }

  // Write output.
  std::error_code ec;
  auto output = std::make_unique<ToolOutputFile>(
      outputFilename, ec, sys::fs::OF_None);
  if (ec) {
    errs() << "Error opening output file: " << ec.message() << "\n";
    return 1;
  }

  llvmModule->print(output->os(), nullptr);
  output->keep();

  if (!quiet)
    errs() << "Compiled successfully to " << outputFilename << "\n";
  return 0;
}

//===- LLVMLoweringPass.cpp - DQC/MPI to LLVM Lowering ----*- C++ -*-===//
//
// Lowers all DQC and MPI dialect operations to LLVM dialect calls
// targeting the DQC runtime library (libdqc_runtime).
//
//===----------------------------------------------------------------===//

#include "dqc/DQCDialect.h"
#include "dqc/DQCOps.h"
#include "dqc/MPIDialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "llvm-lowering"

using namespace mlir;

namespace {

//===--------------------------------------------------------------===//
// Type converter: DQC types -> LLVM i32
//===--------------------------------------------------------------===//

class DQCTypeConverter : public LLVMTypeConverter {
public:
  DQCTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx) {
    addConversion([](dqc::QubitType type) {
      return IntegerType::get(type.getContext(), 32);
    });
    addConversion([](dqc::EPRHandleType type) {
      return IntegerType::get(type.getContext(), 32);
    });
    addConversion([](dqc::CbitType type) {
      return IntegerType::get(type.getContext(), 32);
    });
  }
};

//===--------------------------------------------------------------===//
// Helpers
//===--------------------------------------------------------------===//

static LLVM::LLVMFuncOp getOrInsertFunc(ModuleOp module, StringRef name,
                                         LLVM::LLVMFunctionType fnType) {
  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;
  OpBuilder builder(module.getBodyRegion());
  return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
}

//===--------------------------------------------------------------===//
// Lowering patterns
//===--------------------------------------------------------------===//

/// dqc.alloc_qubit -> call @dqc_alloc_qubit() : () -> i32
struct AllocQubitLowering : public ConversionPattern {
  AllocQubitLowering(TypeConverter &tc, MLIRContext *ctx)
      : ConversionPattern(tc, "dqc.alloc_qubit", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rw) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto i32 = IntegerType::get(rw.getContext(), 32);
    auto fn = getOrInsertFunc(module, "dqc_alloc_qubit",
                              LLVM::LLVMFunctionType::get(i32, {}));
    auto call = rw.create<LLVM::CallOp>(op->getLoc(), fn, ValueRange{});
    rw.replaceOp(op, call.getResults());
    return success();
  }
};

/// Generic single-qubit gate: dqc.{h,x,y,z,s,t} -> call @dqc_X(i32)
struct SingleQubitGateLowering : public ConversionPattern {
  std::string runtimeFn;

  SingleQubitGateLowering(TypeConverter &tc, MLIRContext *ctx,
                          StringRef opName, StringRef fnName)
      : ConversionPattern(tc, opName, 1, ctx), runtimeFn(fnName.str()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rw) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto i32 = IntegerType::get(rw.getContext(), 32);
    auto voidTy = LLVM::LLVMVoidType::get(rw.getContext());
    auto fn = getOrInsertFunc(module, runtimeFn,
                              LLVM::LLVMFunctionType::get(voidTy, {i32}));
    rw.create<LLVM::CallOp>(op->getLoc(), fn, ValueRange{operands[0]});
    rw.eraseOp(op);
    return success();
  }
};

/// Parametric rotation: dqc.{rx,ry,rz} -> call @dqc_rx(i32, f64)
struct RotationGateLowering : public ConversionPattern {
  std::string runtimeFn;

  RotationGateLowering(TypeConverter &tc, MLIRContext *ctx,
                       StringRef opName, StringRef fnName)
      : ConversionPattern(tc, opName, 1, ctx), runtimeFn(fnName.str()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rw) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto i32 = IntegerType::get(rw.getContext(), 32);
    auto f64 = Float64Type::get(rw.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(rw.getContext());
    auto fn = getOrInsertFunc(module, runtimeFn,
                              LLVM::LLVMFunctionType::get(voidTy, {i32, f64}));

    auto angle = op->getAttrOfType<FloatAttr>("angle");
    auto angleVal = rw.create<LLVM::ConstantOp>(op->getLoc(), f64, angle);
    rw.create<LLVM::CallOp>(op->getLoc(), fn,
                             ValueRange{operands[0], angleVal});
    rw.eraseOp(op);
    return success();
  }
};

/// Two-qubit gate: dqc.{cnot,cz,swap} -> call @dqc_X(i32, i32)
struct TwoQubitGateLowering : public ConversionPattern {
  std::string runtimeFn;

  TwoQubitGateLowering(TypeConverter &tc, MLIRContext *ctx,
                       StringRef opName, StringRef fnName)
      : ConversionPattern(tc, opName, 1, ctx), runtimeFn(fnName.str()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rw) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto i32 = IntegerType::get(rw.getContext(), 32);
    auto voidTy = LLVM::LLVMVoidType::get(rw.getContext());
    auto fn = getOrInsertFunc(module, runtimeFn,
                              LLVM::LLVMFunctionType::get(voidTy, {i32, i32}));
    rw.create<LLVM::CallOp>(op->getLoc(), fn,
                             ValueRange{operands[0], operands[1]});
    rw.eraseOp(op);
    return success();
  }
};

/// dqc.ccx -> call @dqc_ccx(i32, i32, i32)
struct ToffoliLowering : public ConversionPattern {
  ToffoliLowering(TypeConverter &tc, MLIRContext *ctx)
      : ConversionPattern(tc, "dqc.ccx", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rw) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto i32 = IntegerType::get(rw.getContext(), 32);
    auto voidTy = LLVM::LLVMVoidType::get(rw.getContext());
    auto fn = getOrInsertFunc(
        module, "dqc_ccx",
        LLVM::LLVMFunctionType::get(voidTy, {i32, i32, i32}));
    rw.create<LLVM::CallOp>(op->getLoc(), fn,
                             ValueRange{operands[0], operands[1], operands[2]});
    rw.eraseOp(op);
    return success();
  }
};

/// dqc.measure -> call @dqc_measure(i32) : i32
struct MeasureLowering : public ConversionPattern {
  MeasureLowering(TypeConverter &tc, MLIRContext *ctx)
      : ConversionPattern(tc, "dqc.measure", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rw) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto i32 = IntegerType::get(rw.getContext(), 32);
    auto fn = getOrInsertFunc(module, "dqc_measure",
                              LLVM::LLVMFunctionType::get(i32, {i32}));
    auto call = rw.create<LLVM::CallOp>(op->getLoc(), fn,
                                         ValueRange{operands[0]});
    rw.replaceOp(op, call.getResults());
    return success();
  }
};

/// dqc.epr_alloc -> stack alloc + call @dqc_distribute_epr + load
struct EPRAllocLLVMLowering : public ConversionPattern {
  EPRAllocLLVMLowering(TypeConverter &tc, MLIRContext *ctx)
      : ConversionPattern(tc, "dqc.epr_alloc", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value>,
                                ConversionPatternRewriter &rw) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();
    auto i32 = IntegerType::get(rw.getContext(), 32);
    auto ptr = LLVM::LLVMPointerType::get(rw.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(rw.getContext());
    auto fn = getOrInsertFunc(
        module, "dqc_distribute_epr",
        LLVM::LLVMFunctionType::get(voidTy, {i32, i32, ptr}));

    auto one = rw.create<LLVM::ConstantOp>(loc, i32, rw.getI32IntegerAttr(1));
    auto alloca = rw.create<LLVM::AllocaOp>(loc, ptr, i32, one);

    auto src = rw.create<LLVM::ConstantOp>(
        loc, i32, op->getAttrOfType<IntegerAttr>("source_qpu"));
    auto tgt = rw.create<LLVM::ConstantOp>(
        loc, i32, op->getAttrOfType<IntegerAttr>("target_qpu"));

    rw.create<LLVM::CallOp>(loc, fn, ValueRange{src, tgt, alloca});
    auto eprId = rw.create<LLVM::LoadOp>(loc, i32, alloca);
    rw.replaceOp(op, eprId.getResult());
    return success();
  }
};

/// mpi.distribute_epr -> same runtime call as epr_alloc
struct MPIDistributeEPRLowering : public ConversionPattern {
  MPIDistributeEPRLowering(TypeConverter &tc, MLIRContext *ctx)
      : ConversionPattern(tc, "mpi.distribute_epr", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value>,
                                ConversionPatternRewriter &rw) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();
    auto i32 = IntegerType::get(rw.getContext(), 32);
    auto ptr = LLVM::LLVMPointerType::get(rw.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(rw.getContext());
    auto fn = getOrInsertFunc(
        module, "dqc_distribute_epr",
        LLVM::LLVMFunctionType::get(voidTy, {i32, i32, ptr}));

    auto one = rw.create<LLVM::ConstantOp>(loc, i32, rw.getI32IntegerAttr(1));
    auto alloca = rw.create<LLVM::AllocaOp>(loc, ptr, i32, one);

    auto srcAttr = op->getAttrOfType<IntegerAttr>("src_qpu");
    if (!srcAttr) srcAttr = op->getAttrOfType<IntegerAttr>("source_qpu");
    auto tgtAttr = op->getAttrOfType<IntegerAttr>("tgt_qpu");
    if (!tgtAttr) tgtAttr = op->getAttrOfType<IntegerAttr>("target_qpu");

    auto src = rw.create<LLVM::ConstantOp>(loc, i32, srcAttr);
    auto tgt = rw.create<LLVM::ConstantOp>(loc, i32, tgtAttr);

    rw.create<LLVM::CallOp>(loc, fn, ValueRange{src, tgt, alloca});
    auto eprId = rw.create<LLVM::LoadOp>(loc, i32, alloca);
    rw.replaceOp(op, eprId.getResult());
    return success();
  }
};

/// Helper to lower telegate ops (both dqc.telegate and mpi.telegate_sequence).
static LogicalResult lowerTeleGate(Operation *op, ArrayRef<Value> operands,
                                   ConversionPatternRewriter &rw) {
  auto module = op->getParentOfType<ModuleOp>();
  auto loc = op->getLoc();
  auto i32 = IntegerType::get(rw.getContext(), 32);
  auto voidTy = LLVM::LLVMVoidType::get(rw.getContext());
  auto fn = getOrInsertFunc(
      module, "dqc_telegate_sequence",
      LLVM::LLVMFunctionType::get(voidTy, {i32, i32, i32, i32, i32}));

  auto ctrlQpu = op->getAttrOfType<IntegerAttr>("control_qpu");
  auto tgtQpu = op->getAttrOfType<IntegerAttr>("target_qpu");
  Value cv = rw.create<LLVM::ConstantOp>(
      loc, i32, ctrlQpu ? ctrlQpu : rw.getI32IntegerAttr(0));
  Value tv = rw.create<LLVM::ConstantOp>(
      loc, i32, tgtQpu ? tgtQpu : rw.getI32IntegerAttr(1));

  rw.create<LLVM::CallOp>(
      loc, fn, ValueRange{operands[0], operands[1], operands[2], cv, tv});
  rw.eraseOp(op);
  return success();
}

struct TeleGateLLVMLowering : public ConversionPattern {
  TeleGateLLVMLowering(TypeConverter &tc, MLIRContext *ctx)
      : ConversionPattern(tc, "dqc.telegate", 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rw) const override {
    return lowerTeleGate(op, operands, rw);
  }
};

struct MPITeleGateLowering : public ConversionPattern {
  MPITeleGateLowering(TypeConverter &tc, MLIRContext *ctx)
      : ConversionPattern(tc, "mpi.telegate_sequence", 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rw) const override {
    return lowerTeleGate(op, operands, rw);
  }
};

/// dqc.mcx -> alloca control array + call @dqc_mcx(ptr, i32, i32)
struct MCXLowering : public ConversionPattern {
  MCXLowering(TypeConverter &tc, MLIRContext *ctx)
      : ConversionPattern(tc, "dqc.mcx", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rw) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();
    auto i32 = IntegerType::get(rw.getContext(), 32);
    auto ptr = LLVM::LLVMPointerType::get(rw.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(rw.getContext());
    auto fn = getOrInsertFunc(
        module, "dqc_mcx",
        LLVM::LLVMFunctionType::get(voidTy, {ptr, i32, i32}));

    int numControls = operands.size() - 1;
    auto numCtrlVal = rw.create<LLVM::ConstantOp>(
        loc, i32, rw.getI32IntegerAttr(numControls));

    // Alloca array for control qubit indices
    auto alloca = rw.create<LLVM::AllocaOp>(loc, ptr, i32, numCtrlVal);

    // Store each control qubit index
    for (int c = 0; c < numControls; c++) {
      auto idx = rw.create<LLVM::ConstantOp>(
          loc, i32, rw.getI32IntegerAttr(c));
      auto gep = rw.create<LLVM::GEPOp>(loc, ptr, i32, alloca, ValueRange{idx});
      rw.create<LLVM::StoreOp>(loc, operands[c], gep);
    }

    // Target is last operand
    rw.create<LLVM::CallOp>(
        loc, fn, ValueRange{alloca, numCtrlVal, operands[numControls]});
    rw.eraseOp(op);
    return success();
  }
};

/// dqc.mcp -> alloca control array + call @dqc_mcp(ptr, i32, i32, f64)
struct MCPLowering : public ConversionPattern {
  MCPLowering(TypeConverter &tc, MLIRContext *ctx)
      : ConversionPattern(tc, "dqc.mcp", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rw) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();
    auto i32 = IntegerType::get(rw.getContext(), 32);
    auto f64 = Float64Type::get(rw.getContext());
    auto ptr = LLVM::LLVMPointerType::get(rw.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(rw.getContext());
    auto fn = getOrInsertFunc(
        module, "dqc_mcp",
        LLVM::LLVMFunctionType::get(voidTy, {ptr, i32, i32, f64}));

    int numControls = operands.size() - 1;
    auto numCtrlVal = rw.create<LLVM::ConstantOp>(
        loc, i32, rw.getI32IntegerAttr(numControls));

    auto alloca = rw.create<LLVM::AllocaOp>(loc, ptr, i32, numCtrlVal);

    for (int c = 0; c < numControls; c++) {
      auto idx = rw.create<LLVM::ConstantOp>(loc, i32, rw.getI32IntegerAttr(c));
      auto gep = rw.create<LLVM::GEPOp>(loc, ptr, i32, alloca, ValueRange{idx});
      rw.create<LLVM::StoreOp>(loc, operands[c], gep);
    }

    auto angle = op->getAttrOfType<FloatAttr>("angle");
    auto angleVal = rw.create<LLVM::ConstantOp>(loc, f64, angle);

    rw.create<LLVM::CallOp>(
        loc, fn, ValueRange{alloca, numCtrlVal, operands[numControls], angleVal});
    rw.eraseOp(op);
    return success();
  }
};

/// dqc.reset -> call @dqc_reset(i32)
struct ResetLowering : public ConversionPattern {
  ResetLowering(TypeConverter &tc, MLIRContext *ctx)
      : ConversionPattern(tc, "dqc.reset", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rw) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto i32 = IntegerType::get(rw.getContext(), 32);
    auto voidTy = LLVM::LLVMVoidType::get(rw.getContext());
    auto fn = getOrInsertFunc(module, "dqc_reset",
                              LLVM::LLVMFunctionType::get(voidTy, {i32}));
    rw.create<LLVM::CallOp>(op->getLoc(), fn, ValueRange{operands[0]});
    rw.eraseOp(op);
    return success();
  }
};

/// dqc.repeat N -> loop with counter (header/body/merge blocks)
struct RepeatLowering : public ConversionPattern {
  RepeatLowering(TypeConverter &tc, MLIRContext *ctx)
      : ConversionPattern(tc, "dqc.repeat", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value>,
                                ConversionPatternRewriter &rw) const override {
    auto loc = op->getLoc();
    auto i64 = IntegerType::get(rw.getContext(), 64);
    auto i1 = IntegerType::get(rw.getContext(), 1);

    int64_t count = op->getAttrOfType<IntegerAttr>("count").getInt();
    auto limitVal = rw.create<LLVM::ConstantOp>(
        loc, i64, rw.getI64IntegerAttr(count));
    auto zeroVal = rw.create<LLVM::ConstantOp>(
        loc, i64, rw.getI64IntegerAttr(0));
    auto oneVal = rw.create<LLVM::ConstantOp>(
        loc, i64, rw.getI64IntegerAttr(1));

    Block *currentBlock = rw.getInsertionBlock();
    Block *mergeBlock = rw.splitBlock(currentBlock, rw.getInsertionPoint());

    // Header block: check counter < limit
    Block *headerBlock = rw.createBlock(mergeBlock);
    headerBlock->addArgument(i64, loc);

    // Body block: clone ops + increment counter
    Block *bodyBlock = rw.createBlock(mergeBlock);

    // Wire current -> header (counter = 0)
    rw.setInsertionPointToEnd(currentBlock);
    rw.create<LLVM::BrOp>(loc, ValueRange{zeroVal}, headerBlock);

    // Header: if counter < limit goto body else goto merge
    rw.setInsertionPointToStart(headerBlock);
    Value counter = headerBlock->getArgument(0);
    auto cmp = rw.create<LLVM::ICmpOp>(
        loc, i1, LLVM::ICmpPredicate::slt, counter, limitVal);
    rw.create<LLVM::CondBrOp>(loc, cmp, bodyBlock, mergeBlock);

    // Body: clone ops + increment + branch to header
    rw.setInsertionPointToStart(bodyBlock);
    auto &bodyOps = op->getRegion(0).front().getOperations();
    for (auto &bodyOp : bodyOps) {
      rw.clone(bodyOp);
    }
    auto nextCounter = rw.create<LLVM::AddOp>(loc, i64, counter, oneVal);
    rw.create<LLVM::BrOp>(loc, ValueRange{nextCounter}, headerBlock);

    rw.eraseOp(op);
    return success();
  }
};

/// dqc.c_if -> icmp + cond_br with then/merge blocks
struct CondLowering : public ConversionPattern {
  CondLowering(TypeConverter &tc, MLIRContext *ctx)
      : ConversionPattern(tc, "dqc.c_if", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rw) const override {
    auto loc = op->getLoc();
    auto i32 = IntegerType::get(rw.getContext(), 32);
    auto i1 = IntegerType::get(rw.getContext(), 1);

    // Compare cbit == 1
    auto one = rw.create<LLVM::ConstantOp>(loc, i32, rw.getI32IntegerAttr(1));
    auto cmp = rw.create<LLVM::ICmpOp>(
        loc, i1, LLVM::ICmpPredicate::eq, operands[0], one);

    // Split current block: everything after c_if goes to merge
    Block *currentBlock = rw.getInsertionBlock();
    Block *mergeBlock = rw.splitBlock(currentBlock, rw.getInsertionPoint());

    // Create then block
    Block *thenBlock = rw.createBlock(mergeBlock);

    // Clone body ops into then block
    auto &bodyOps = op->getRegion(0).front().getOperations();
    rw.setInsertionPointToStart(thenBlock);
    for (auto &bodyOp : bodyOps) {
      rw.clone(bodyOp);
    }
    rw.create<LLVM::BrOp>(loc, ValueRange{}, mergeBlock);

    // Add conditional branch at end of current block
    rw.setInsertionPointToEnd(currentBlock);
    rw.create<LLVM::CondBrOp>(loc, cmp, thenBlock, mergeBlock);

    rw.eraseOp(op);
    return success();
  }
};

/// Erase metadata-only ops that have no runtime semantics.
struct EraseOpLowering : public ConversionPattern {
  EraseOpLowering(TypeConverter &tc, MLIRContext *ctx, StringRef opName)
      : ConversionPattern(tc, opName, 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value>,
                                ConversionPatternRewriter &rw) const override {
    rw.eraseOp(op);
    return success();
  }
};

//===--------------------------------------------------------------===//
// Pass
//===--------------------------------------------------------------===//

class LLVMLoweringPass
    : public PassWrapper<LLVMLoweringPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "dqc-llvm-lowering"; }
  StringRef getDescription() const final {
    return "Lower DQC/MPI dialects to LLVM dialect (runtime calls)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<dqc::mpi::MPIDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    // Count qubits per function before conversion erases DQC ops.
    // Falls back to the dqc.partition attribute when alloc_qubit ops
    // have been DCE'd by earlier passes.
    llvm::StringMap<int> qubitCounts;
    module.walk([&](func::FuncOp funcOp) {
      int n = 0;
      funcOp.walk([&](Operation *op) {
        if (op->getName().getStringRef() == "dqc.alloc_qubit")
          n++;
      });
      if (n == 0) {
        if (auto p = funcOp->getAttrOfType<DictionaryAttr>("dqc.partition"))
          n = p.size();
      }
      if (n > 0)
        qubitCounts[funcOp.getSymName()] = n;
    });

    // Configure conversion.
    DQCTypeConverter typeConverter(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect<dqc::DQCDialect>();
    target.addIllegalDialect<dqc::mpi::MPIDialect>();
    target.addIllegalDialect<func::FuncDialect>();

    // Register patterns.
    RewritePatternSet patterns(ctx);

    patterns.add<AllocQubitLowering>(typeConverter, ctx);
    patterns.add<SingleQubitGateLowering>(typeConverter, ctx, "dqc.h", "dqc_h");
    patterns.add<SingleQubitGateLowering>(typeConverter, ctx, "dqc.x", "dqc_x");
    patterns.add<SingleQubitGateLowering>(typeConverter, ctx, "dqc.y", "dqc_y");
    patterns.add<SingleQubitGateLowering>(typeConverter, ctx, "dqc.z", "dqc_z");
    patterns.add<SingleQubitGateLowering>(typeConverter, ctx, "dqc.s", "dqc_s");
    patterns.add<SingleQubitGateLowering>(typeConverter, ctx, "dqc.t", "dqc_t");
    patterns.add<SingleQubitGateLowering>(typeConverter, ctx,
                                          "dqc.local_gate", "dqc_h");
    patterns.add<RotationGateLowering>(typeConverter, ctx, "dqc.rx", "dqc_rx");
    patterns.add<RotationGateLowering>(typeConverter, ctx, "dqc.ry", "dqc_ry");
    patterns.add<RotationGateLowering>(typeConverter, ctx, "dqc.rz", "dqc_rz");
    patterns.add<TwoQubitGateLowering>(typeConverter, ctx,
                                       "dqc.cnot", "dqc_cnot");
    patterns.add<TwoQubitGateLowering>(typeConverter, ctx, "dqc.cz", "dqc_cz");
    patterns.add<TwoQubitGateLowering>(typeConverter, ctx,
                                       "dqc.swap", "dqc_swap");
    patterns.add<ToffoliLowering>(typeConverter, ctx);
    patterns.add<MeasureLowering>(typeConverter, ctx);
    patterns.add<ResetLowering>(typeConverter, ctx);
    patterns.add<MCXLowering>(typeConverter, ctx);
    patterns.add<MCPLowering>(typeConverter, ctx);
    patterns.add<CondLowering>(typeConverter, ctx);
    patterns.add<RepeatLowering>(typeConverter, ctx);
    patterns.add<EPRAllocLLVMLowering>(typeConverter, ctx);
    patterns.add<TeleGateLLVMLowering>(typeConverter, ctx);
    patterns.add<MPIDistributeEPRLowering>(typeConverter, ctx);
    patterns.add<MPITeleGateLowering>(typeConverter, ctx);
    patterns.add<EraseOpLowering>(typeConverter, ctx, "dqc.partition_info");
    patterns.add<EraseOpLowering>(typeConverter, ctx, "dqc.epr_consume");
    patterns.add<EraseOpLowering>(typeConverter, ctx, "dqc.barrier");

    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Inject runtime init/teardown calls.
    auto i32Ty = IntegerType::get(ctx, 32);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    module.walk([&](LLVM::LLVMFuncOp funcOp) {
      auto it = qubitCounts.find(funcOp.getSymName());
      if (it == qubitCounts.end())
        return;
      int numQubits = it->second;
      auto loc = funcOp.getLoc();

      // dqc_init(numQubits) at entry.
      {
        OpBuilder b(ctx);
        b.setInsertionPointToStart(&funcOp.getBody().front());
        auto fn = getOrInsertFunc(
            module, "dqc_init",
            LLVM::LLVMFunctionType::get(voidTy, {i32Ty}));
        auto nv = b.create<LLVM::ConstantOp>(
            loc, i32Ty, b.getI32IntegerAttr(numQubits));
        b.create<LLVM::CallOp>(loc, fn, ValueRange{nv});
      }

      // dqc_dump_state() + dqc_finalize() before each return.
      SmallVector<LLVM::ReturnOp> returns;
      funcOp.walk([&](LLVM::ReturnOp ret) { returns.push_back(ret); });

      auto voidFnTy = LLVM::LLVMFunctionType::get(voidTy, {});
      auto dumpFn = getOrInsertFunc(module, "dqc_dump_state", voidFnTy);
      auto finFn = getOrInsertFunc(module, "dqc_finalize", voidFnTy);

      for (auto ret : returns) {
        OpBuilder b(ret);
        b.create<LLVM::CallOp>(ret.getLoc(), dumpFn, ValueRange{});
        b.create<LLVM::CallOp>(ret.getLoc(), finFn, ValueRange{});
      }
    });
  }
};

} // anonymous namespace

namespace dqc {

std::unique_ptr<mlir::Pass> createLLVMLoweringPass() {
  return std::make_unique<::LLVMLoweringPass>();
}

} // namespace dqc

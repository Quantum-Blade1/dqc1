//===- DQCOps.cpp - DQC Op Implementations ---*- C++ -*-===//
//
// This file implements operations in the DQC dialect.
//
//===----------------------------------------------------===//

#include "dqc/DQCOps.h"
#include "dqc/DQCDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace llvm;
using namespace mlir;
using namespace dqc;

#define GET_OP_CLASSES
#include "dqc/DQCOps.cpp.inc"

//===------------------------------------------------------===//
// DQCTeleGateMultiOp: Custom Assembly Format
//===------------------------------------------------------===//

ParseResult TeleGateMultiOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  SmallVector<OpAsmParser::UnresolvedOperand, 8> operands;
  SmallVector<Type, 8> operand_types;
  SmallVector<int32_t, 4> target_qpus;

  // Parse: control_qubit, target_qubits..., epr_handle
  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::None))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.parseColon())
    return failure();

  if (parser.parseTypeList(operand_types))
    return failure();

  if (parser.parseArrow())
    return failure();

  SmallVector<Type, 4> result_types;
  if (parser.parseTypeList(result_types))
    return failure();

  // Resolve operands
  if (parser.resolveOperands(operands, TypeRange(operand_types), loc, result.operands))
    return failure();

  // Add types
  result.addTypes(result_types);

  return success();
}

void TeleGateMultiOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printOperands(getOperands());
  printer << " : ";
  llvm::interleaveComma(getOperandTypes(), printer, [&](Type t) { printer.printType(t); });
  printer << " -> ";
  llvm::interleaveComma(getResultTypes(), printer, [&](Type t) { printer.printType(t); });
}

//===------------------------------------------------------===//
// MCXOp: Custom Assembly Format
// Syntax: dqc.mcx %c0, %c1, ..., %target : (!dqc.qubit, ...)
//===------------------------------------------------------===//

ParseResult MCXOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 8> operands;
  SmallVector<Type, 8> types;
  llvm::SMLoc loc = parser.getCurrentLocation();

  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::None))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();
  if (parser.parseLParen())
    return failure();
  if (parser.parseTypeList(types))
    return failure();
  if (parser.parseRParen())
    return failure();

  if (parser.resolveOperands(operands, types, loc, result.operands))
    return failure();
  return success();
}

void MCXOp::print(OpAsmPrinter &printer) {
  printer << " ";
  llvm::interleaveComma(getOperands(), printer,
                        [&](Value v) { printer.printOperand(v); });
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : (";
  llvm::interleaveComma(getOperandTypes(), printer,
                        [&](Type t) { printer.printType(t); });
  printer << ")";
}

//===------------------------------------------------------===//
// CondOp (c_if): Custom Assembly Format
// Syntax: dqc.c_if %cbit { body }
//===------------------------------------------------------===//

ParseResult CondOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand cond;
  Type condType;

  if (parser.parseOperand(cond))
    return failure();

  condType = CbitType::get(parser.getContext());
  if (parser.resolveOperand(cond, condType, result.operands))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {}, {}))
    return failure();
  if (body->empty())
    body->emplaceBlock();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void CondOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printOperand(getCondition());
  printer << " ";
  printer.printRegion(getBody(), false);
  printer.printOptionalAttrDict((*this)->getAttrs());
}

//===------------------------------------------------------===//
// RepeatOp: Custom Assembly Format
// Syntax: dqc.repeat 3 { body }
//===------------------------------------------------------===//

ParseResult RepeatOp::parse(OpAsmParser &parser, OperationState &result) {
  int64_t count;
  if (parser.parseInteger(count))
    return failure();
  result.addAttribute("count",
      IntegerAttr::get(IntegerType::get(parser.getContext(), 64), count));

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {}, {}))
    return failure();
  if (body->empty())
    body->emplaceBlock();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void RepeatOp::print(OpAsmPrinter &printer) {
  printer << " " << getCount();
  printer << " ";
  printer.printRegion(getBody(), false);
  printer.printOptionalAttrDict((*this)->getAttrs(), {"count"});
}

//===------------------------------------------------------===//
// MCPOp: Custom Assembly Format
// Syntax: dqc.mcp %c0, ..., %target angle : (!dqc.qubit, ...)
//===------------------------------------------------------===//

ParseResult MCPOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 8> operands;
  SmallVector<Type, 8> types;
  llvm::SMLoc loc = parser.getCurrentLocation();

  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::None))
    return failure();

  double angle;
  if (parser.parseFloat(angle))
    return failure();
  result.addAttribute("angle",
      FloatAttr::get(Float64Type::get(parser.getContext()), angle));

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();
  if (parser.parseLParen())
    return failure();
  if (parser.parseTypeList(types))
    return failure();
  if (parser.parseRParen())
    return failure();

  if (parser.resolveOperands(operands, types, loc, result.operands))
    return failure();
  return success();
}

void MCPOp::print(OpAsmPrinter &printer) {
  printer << " ";
  llvm::interleaveComma(getOperands(), printer,
                        [&](Value v) { printer.printOperand(v); });
  printer << " ";
  printer << (*this)->getAttrOfType<FloatAttr>("angle").getValueAsDouble();
  printer.printOptionalAttrDict((*this)->getAttrs(), {"angle"});
  printer << " : (";
  llvm::interleaveComma(getOperandTypes(), printer,
                        [&](Type t) { printer.printType(t); });
  printer << ")";
}

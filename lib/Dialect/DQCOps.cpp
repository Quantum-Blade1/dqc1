//===- DQCOps.cpp - DQC Op Implementations ---*- C++ -*-===//
//
// This file implements operations in the DQC dialect.
//
//===----------------------------------------------------===//

#include "dqc/DQCOps.h"
#include "dqc/DQCDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace dqc;

#define GET_OP_CLASSES
#include "dqc/DQCOps.cpp.inc"

//===------------------------------------------------------===//
// DQCTeleGateMultiOp: Custom Assembly Format
//===------------------------------------------------------===//

ParseResult DQCTeleGateMultiOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  Location loc = parser.getCurrentLocation();
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

  return success();
}

void DQCTeleGateMultiOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printOperands(getOperands());
  printer << " : ";
  printer.printTypes(getOperandTypes());
  printer << " -> ";
  printer.printTypes(getResultTypes());
}


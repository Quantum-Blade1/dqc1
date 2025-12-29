/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::dqc::EPRAllocOp,
::dqc::EPRConsumeOp,
::dqc::PartitionInfoOp,
::dqc::TeleGateMultiOp,
::dqc::TeleGateOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace dqc {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_DQCDialect0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((llvm::isa<dqc::EPRHandleType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be epr_handle, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_DQCDialect1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((llvm::isa<dqc::QubitType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be qubit, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_DQCDialect0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_DQCDialect1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::FloatAttr>())) && ((attr.cast<::mlir::FloatAttr>().getType().isF32())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit float attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_DQCDialect2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::DictionaryAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: dictionary of named attribute values";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_DQCDialect3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit integer array attribute";
  }
  return ::mlir::success();
}
} // namespace dqc
namespace dqc {

//===----------------------------------------------------------------------===//
// ::dqc::EPRAllocOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
EPRAllocOpGenericAdaptorBase::EPRAllocOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("dqc.epr_alloc", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> EPRAllocOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr EPRAllocOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr EPRAllocOpGenericAdaptorBase::getSourceQpuAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, EPRAllocOp::getSourceQpuAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t EPRAllocOpGenericAdaptorBase::getSourceQpu() {
  auto attr = getSourceQpuAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr EPRAllocOpGenericAdaptorBase::getTargetQpuAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, EPRAllocOp::getTargetQpuAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t EPRAllocOpGenericAdaptorBase::getTargetQpu() {
  auto attr = getTargetQpuAttr();
  return attr.getValue().getZExtValue();
}

::mlir::FloatAttr EPRAllocOpGenericAdaptorBase::getFidelityAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, EPRAllocOp::getFidelityAttrName(*odsOpName)).dyn_cast_or_null<::mlir::FloatAttr>();
  return attr;
}

::std::optional< ::llvm::APFloat > EPRAllocOpGenericAdaptorBase::getFidelity() {
  auto attr = getFidelityAttr();
  return attr ? ::std::optional< ::llvm::APFloat >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
EPRAllocOpAdaptor::EPRAllocOpAdaptor(EPRAllocOp op) : EPRAllocOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult EPRAllocOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_source_qpu;
  ::mlir::Attribute tblgen_fidelity;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'dqc.epr_alloc' op ""requires attribute 'source_qpu'");
    if (namedAttrIt->getName() == EPRAllocOp::getSourceQpuAttrName(*odsOpName)) {
      tblgen_source_qpu = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == EPRAllocOp::getFidelityAttrName(*odsOpName)) {
      tblgen_fidelity = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_target_qpu;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'dqc.epr_alloc' op ""requires attribute 'target_qpu'");
    if (namedAttrIt->getName() == EPRAllocOp::getTargetQpuAttrName(*odsOpName)) {
      tblgen_target_qpu = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_source_qpu && !(((tblgen_source_qpu.isa<::mlir::IntegerAttr>())) && ((tblgen_source_qpu.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'dqc.epr_alloc' op ""attribute 'source_qpu' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_target_qpu && !(((tblgen_target_qpu.isa<::mlir::IntegerAttr>())) && ((tblgen_target_qpu.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'dqc.epr_alloc' op ""attribute 'target_qpu' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_fidelity && !(((tblgen_fidelity.isa<::mlir::FloatAttr>())) && ((tblgen_fidelity.cast<::mlir::FloatAttr>().getType().isF32()))))
    return emitError(loc, "'dqc.epr_alloc' op ""attribute 'fidelity' failed to satisfy constraint: 32-bit float attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> EPRAllocOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range EPRAllocOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> EPRAllocOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range EPRAllocOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EPRAllocOp::getEprHandle() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr EPRAllocOp::getSourceQpuAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getSourceQpuAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t EPRAllocOp::getSourceQpu() {
  auto attr = getSourceQpuAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr EPRAllocOp::getTargetQpuAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTargetQpuAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t EPRAllocOp::getTargetQpu() {
  auto attr = getTargetQpuAttr();
  return attr.getValue().getZExtValue();
}

::mlir::FloatAttr EPRAllocOp::getFidelityAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getFidelityAttrName()).dyn_cast_or_null<::mlir::FloatAttr>();
}

::std::optional< ::llvm::APFloat > EPRAllocOp::getFidelity() {
  auto attr = getFidelityAttr();
  return attr ? ::std::optional< ::llvm::APFloat >(attr.getValue()) : (::std::nullopt);
}

void EPRAllocOp::setSourceQpuAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getSourceQpuAttrName(), attr);
}

void EPRAllocOp::setSourceQpu(uint32_t attrValue) {
  (*this)->setAttr(getSourceQpuAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void EPRAllocOp::setTargetQpuAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getTargetQpuAttrName(), attr);
}

void EPRAllocOp::setTargetQpu(uint32_t attrValue) {
  (*this)->setAttr(getTargetQpuAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void EPRAllocOp::setFidelityAttr(::mlir::FloatAttr attr) {
  (*this)->setAttr(getFidelityAttrName(), attr);
}

void EPRAllocOp::setFidelity(::std::optional<::llvm::APFloat> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getFidelityAttrName(), ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), *attrValue));
    (*this)->removeAttr(getFidelityAttrName());
}

::mlir::Attribute EPRAllocOp::removeFidelityAttr() {
  return (*this)->removeAttr(getFidelityAttrName());
}

void EPRAllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type epr_handle, ::mlir::IntegerAttr source_qpu, ::mlir::IntegerAttr target_qpu, /*optional*/::mlir::FloatAttr fidelity) {
  odsState.addAttribute(getSourceQpuAttrName(odsState.name), source_qpu);
  odsState.addAttribute(getTargetQpuAttrName(odsState.name), target_qpu);
  if (fidelity) {
    odsState.addAttribute(getFidelityAttrName(odsState.name), fidelity);
  }
  odsState.addTypes(epr_handle);
}

void EPRAllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr source_qpu, ::mlir::IntegerAttr target_qpu, /*optional*/::mlir::FloatAttr fidelity) {
  odsState.addAttribute(getSourceQpuAttrName(odsState.name), source_qpu);
  odsState.addAttribute(getTargetQpuAttrName(odsState.name), target_qpu);
  if (fidelity) {
    odsState.addAttribute(getFidelityAttrName(odsState.name), fidelity);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EPRAllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type epr_handle, uint32_t source_qpu, uint32_t target_qpu, /*optional*/::mlir::FloatAttr fidelity) {
  odsState.addAttribute(getSourceQpuAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), source_qpu));
  odsState.addAttribute(getTargetQpuAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), target_qpu));
  if (fidelity) {
    odsState.addAttribute(getFidelityAttrName(odsState.name), fidelity);
  }
  odsState.addTypes(epr_handle);
}

void EPRAllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t source_qpu, uint32_t target_qpu, /*optional*/::mlir::FloatAttr fidelity) {
  odsState.addAttribute(getSourceQpuAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), source_qpu));
  odsState.addAttribute(getTargetQpuAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), target_qpu));
  if (fidelity) {
    odsState.addAttribute(getFidelityAttrName(odsState.name), fidelity);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EPRAllocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult EPRAllocOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_source_qpu;
  ::mlir::Attribute tblgen_fidelity;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'source_qpu'");
    if (namedAttrIt->getName() == getSourceQpuAttrName()) {
      tblgen_source_qpu = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getFidelityAttrName()) {
      tblgen_fidelity = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_target_qpu;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'target_qpu'");
    if (namedAttrIt->getName() == getTargetQpuAttrName()) {
      tblgen_target_qpu = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_DQCDialect0(*this, tblgen_source_qpu, "source_qpu")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_DQCDialect0(*this, tblgen_target_qpu, "target_qpu")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_DQCDialect1(*this, tblgen_fidelity, "fidelity")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult EPRAllocOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult EPRAllocOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr source_qpuAttr;
  ::mlir::IntegerAttr target_qpuAttr;
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  if (parser.parseCustomAttributeWithFallback(source_qpuAttr, parser.getBuilder().getIntegerType(32), "source_qpu",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(target_qpuAttr, parser.getBuilder().getIntegerType(32), "target_qpu",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  return ::mlir::success();
}

void EPRAllocOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getSourceQpuAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getTargetQpuAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("source_qpu");
  elidedAttrs.push_back("target_qpu");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void EPRAllocOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace dqc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::dqc::EPRAllocOp)

namespace dqc {

//===----------------------------------------------------------------------===//
// ::dqc::EPRConsumeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
EPRConsumeOpGenericAdaptorBase::EPRConsumeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("dqc.epr_consume", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> EPRConsumeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr EPRConsumeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
EPRConsumeOpAdaptor::EPRConsumeOpAdaptor(EPRConsumeOp op) : EPRConsumeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult EPRConsumeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> EPRConsumeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range EPRConsumeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EPRConsumeOp::getEprHandle() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange EPRConsumeOp::getEprHandleMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> EPRConsumeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range EPRConsumeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void EPRConsumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value epr_handle) {
  odsState.addOperands(epr_handle);
}

void EPRConsumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value epr_handle) {
  odsState.addOperands(epr_handle);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EPRConsumeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult EPRConsumeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult EPRConsumeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult EPRConsumeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand epr_handleRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> epr_handleOperands(epr_handleRawOperands);  ::llvm::SMLoc epr_handleOperandsLoc;
  (void)epr_handleOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allOperandTypes;

  epr_handleOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(epr_handleRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(epr_handleOperands, allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void EPRConsumeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getEprHandle();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperandTypes();
}

} // namespace dqc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::dqc::EPRConsumeOp)

namespace dqc {

//===----------------------------------------------------------------------===//
// ::dqc::PartitionInfoOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PartitionInfoOpGenericAdaptorBase::PartitionInfoOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("dqc.partition_info", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PartitionInfoOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr PartitionInfoOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DictionaryAttr PartitionInfoOpGenericAdaptorBase::getPartitionMapAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PartitionInfoOp::getPartitionMapAttrName(*odsOpName)).cast<::mlir::DictionaryAttr>();
  return attr;
}

::mlir::DictionaryAttr PartitionInfoOpGenericAdaptorBase::getPartitionMap() {
  auto attr = getPartitionMapAttr();
  return attr;
}

} // namespace detail
PartitionInfoOpAdaptor::PartitionInfoOpAdaptor(PartitionInfoOp op) : PartitionInfoOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PartitionInfoOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_partition_map;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'dqc.partition_info' op ""requires attribute 'partition_map'");
    if (namedAttrIt->getName() == PartitionInfoOp::getPartitionMapAttrName(*odsOpName)) {
      tblgen_partition_map = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_partition_map && !((tblgen_partition_map.isa<::mlir::DictionaryAttr>())))
    return emitError(loc, "'dqc.partition_info' op ""attribute 'partition_map' failed to satisfy constraint: dictionary of named attribute values");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PartitionInfoOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PartitionInfoOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> PartitionInfoOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PartitionInfoOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr PartitionInfoOp::getPartitionMapAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPartitionMapAttrName()).cast<::mlir::DictionaryAttr>();
}

::mlir::DictionaryAttr PartitionInfoOp::getPartitionMap() {
  auto attr = getPartitionMapAttr();
  return attr;
}

void PartitionInfoOp::setPartitionMapAttr(::mlir::DictionaryAttr attr) {
  (*this)->setAttr(getPartitionMapAttrName(), attr);
}

void PartitionInfoOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::DictionaryAttr partition_map) {
  odsState.addAttribute(getPartitionMapAttrName(odsState.name), partition_map);
}

void PartitionInfoOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DictionaryAttr partition_map) {
  odsState.addAttribute(getPartitionMapAttrName(odsState.name), partition_map);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PartitionInfoOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PartitionInfoOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_partition_map;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'partition_map'");
    if (namedAttrIt->getName() == getPartitionMapAttrName()) {
      tblgen_partition_map = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_DQCDialect2(*this, tblgen_partition_map, "partition_map")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult PartitionInfoOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PartitionInfoOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void PartitionInfoOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void PartitionInfoOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace dqc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::dqc::PartitionInfoOp)

namespace dqc {

//===----------------------------------------------------------------------===//
// ::dqc::TeleGateMultiOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TeleGateMultiOpGenericAdaptorBase::TeleGateMultiOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("dqc.telegate_multi", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TeleGateMultiOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr TeleGateMultiOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr TeleGateMultiOpGenericAdaptorBase::getControlQpuAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TeleGateMultiOp::getControlQpuAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t TeleGateMultiOpGenericAdaptorBase::getControlQpu() {
  auto attr = getControlQpuAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr TeleGateMultiOpGenericAdaptorBase::getTargetQpusAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, TeleGateMultiOp::getTargetQpusAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr TeleGateMultiOpGenericAdaptorBase::getTargetQpus() {
  auto attr = getTargetQpusAttr();
  return attr;
}

} // namespace detail
TeleGateMultiOpAdaptor::TeleGateMultiOpAdaptor(TeleGateMultiOp op) : TeleGateMultiOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TeleGateMultiOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_control_qpu;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'dqc.telegate_multi' op ""requires attribute 'control_qpu'");
    if (namedAttrIt->getName() == TeleGateMultiOp::getControlQpuAttrName(*odsOpName)) {
      tblgen_control_qpu = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_target_qpus;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'dqc.telegate_multi' op ""requires attribute 'target_qpus'");
    if (namedAttrIt->getName() == TeleGateMultiOp::getTargetQpusAttrName(*odsOpName)) {
      tblgen_target_qpus = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_control_qpu && !(((tblgen_control_qpu.isa<::mlir::IntegerAttr>())) && ((tblgen_control_qpu.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'dqc.telegate_multi' op ""attribute 'control_qpu' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_target_qpus && !(((tblgen_target_qpus.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_target_qpus.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))); }))))
    return emitError(loc, "'dqc.telegate_multi' op ""attribute 'target_qpus' failed to satisfy constraint: 32-bit integer array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TeleGateMultiOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range TeleGateMultiOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TeleGateMultiOp::getControlQubit() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range TeleGateMultiOp::getTargetQubits() {
  return getODSOperands(1);
}

::mlir::Value TeleGateMultiOp::getEprHandle() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange TeleGateMultiOp::getControlQubitMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange TeleGateMultiOp::getTargetQubitsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange TeleGateMultiOp::getEprHandleMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TeleGateMultiOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range TeleGateMultiOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range TeleGateMultiOp::getResults() {
  return getODSResults(0);
}

::mlir::IntegerAttr TeleGateMultiOp::getControlQpuAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getControlQpuAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t TeleGateMultiOp::getControlQpu() {
  auto attr = getControlQpuAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr TeleGateMultiOp::getTargetQpusAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTargetQpusAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TeleGateMultiOp::getTargetQpus() {
  auto attr = getTargetQpusAttr();
  return attr;
}

void TeleGateMultiOp::setControlQpuAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getControlQpuAttrName(), attr);
}

void TeleGateMultiOp::setControlQpu(uint32_t attrValue) {
  (*this)->setAttr(getControlQpuAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void TeleGateMultiOp::setTargetQpusAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTargetQpusAttrName(), attr);
}

void TeleGateMultiOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::Value control_qubit, ::mlir::ValueRange target_qubits, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::ArrayAttr target_qpus) {
  odsState.addOperands(control_qubit);
  odsState.addOperands(target_qubits);
  odsState.addOperands(epr_handle);
  odsState.addAttribute(getControlQpuAttrName(odsState.name), control_qpu);
  odsState.addAttribute(getTargetQpusAttrName(odsState.name), target_qpus);
  odsState.addTypes(results);
}

void TeleGateMultiOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::Value control_qubit, ::mlir::ValueRange target_qubits, ::mlir::Value epr_handle, uint32_t control_qpu, ::mlir::ArrayAttr target_qpus) {
  odsState.addOperands(control_qubit);
  odsState.addOperands(target_qubits);
  odsState.addOperands(epr_handle);
  odsState.addAttribute(getControlQpuAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), control_qpu));
  odsState.addAttribute(getTargetQpusAttrName(odsState.name), target_qpus);
  odsState.addTypes(results);
}

void TeleGateMultiOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TeleGateMultiOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_control_qpu;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'control_qpu'");
    if (namedAttrIt->getName() == getControlQpuAttrName()) {
      tblgen_control_qpu = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_target_qpus;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'target_qpus'");
    if (namedAttrIt->getName() == getTargetQpusAttrName()) {
      tblgen_target_qpus = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_DQCDialect0(*this, tblgen_control_qpu, "control_qpu")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_DQCDialect3(*this, tblgen_target_qpus, "target_qpus")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TeleGateMultiOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace dqc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::dqc::TeleGateMultiOp)

namespace dqc {

//===----------------------------------------------------------------------===//
// ::dqc::TeleGateOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TeleGateOpGenericAdaptorBase::TeleGateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("dqc.telegate", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TeleGateOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TeleGateOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr TeleGateOpGenericAdaptorBase::getControlQpuAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TeleGateOp::getControlQpuAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t TeleGateOpGenericAdaptorBase::getControlQpu() {
  auto attr = getControlQpuAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr TeleGateOpGenericAdaptorBase::getTargetQpuAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, TeleGateOp::getTargetQpuAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t TeleGateOpGenericAdaptorBase::getTargetQpu() {
  auto attr = getTargetQpuAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
TeleGateOpAdaptor::TeleGateOpAdaptor(TeleGateOp op) : TeleGateOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TeleGateOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_control_qpu;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'dqc.telegate' op ""requires attribute 'control_qpu'");
    if (namedAttrIt->getName() == TeleGateOp::getControlQpuAttrName(*odsOpName)) {
      tblgen_control_qpu = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_target_qpu;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'dqc.telegate' op ""requires attribute 'target_qpu'");
    if (namedAttrIt->getName() == TeleGateOp::getTargetQpuAttrName(*odsOpName)) {
      tblgen_target_qpu = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_control_qpu && !(((tblgen_control_qpu.isa<::mlir::IntegerAttr>())) && ((tblgen_control_qpu.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'dqc.telegate' op ""attribute 'control_qpu' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_target_qpu && !(((tblgen_target_qpu.isa<::mlir::IntegerAttr>())) && ((tblgen_target_qpu.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'dqc.telegate' op ""attribute 'target_qpu' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TeleGateOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TeleGateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TeleGateOp::getControlQubit() {
  return *getODSOperands(0).begin();
}

::mlir::Value TeleGateOp::getTargetQubit() {
  return *getODSOperands(1).begin();
}

::mlir::Value TeleGateOp::getEprHandle() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange TeleGateOp::getControlQubitMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange TeleGateOp::getTargetQubitMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange TeleGateOp::getEprHandleMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TeleGateOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TeleGateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TeleGateOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr TeleGateOp::getControlQpuAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getControlQpuAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t TeleGateOp::getControlQpu() {
  auto attr = getControlQpuAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr TeleGateOp::getTargetQpuAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTargetQpuAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t TeleGateOp::getTargetQpu() {
  auto attr = getTargetQpuAttr();
  return attr.getValue().getZExtValue();
}

void TeleGateOp::setControlQpuAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getControlQpuAttrName(), attr);
}

void TeleGateOp::setControlQpu(uint32_t attrValue) {
  (*this)->setAttr(getControlQpuAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void TeleGateOp::setTargetQpuAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getTargetQpuAttrName(), attr);
}

void TeleGateOp::setTargetQpu(uint32_t attrValue) {
  (*this)->setAttr(getTargetQpuAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void TeleGateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu) {
  odsState.addOperands(control_qubit);
  odsState.addOperands(target_qubit);
  odsState.addOperands(epr_handle);
  odsState.addAttribute(getControlQpuAttrName(odsState.name), control_qpu);
  odsState.addAttribute(getTargetQpuAttrName(odsState.name), target_qpu);
  odsState.addTypes(result);
}

void TeleGateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu) {
  odsState.addOperands(control_qubit);
  odsState.addOperands(target_qubit);
  odsState.addOperands(epr_handle);
  odsState.addAttribute(getControlQpuAttrName(odsState.name), control_qpu);
  odsState.addAttribute(getTargetQpuAttrName(odsState.name), target_qpu);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TeleGateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu) {
  odsState.addOperands(control_qubit);
  odsState.addOperands(target_qubit);
  odsState.addOperands(epr_handle);
  odsState.addAttribute(getControlQpuAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), control_qpu));
  odsState.addAttribute(getTargetQpuAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), target_qpu));
  odsState.addTypes(result);
}

void TeleGateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu) {
  odsState.addOperands(control_qubit);
  odsState.addOperands(target_qubit);
  odsState.addOperands(epr_handle);
  odsState.addAttribute(getControlQpuAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), control_qpu));
  odsState.addAttribute(getTargetQpuAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), target_qpu));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TeleGateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TeleGateOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_control_qpu;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'control_qpu'");
    if (namedAttrIt->getName() == getControlQpuAttrName()) {
      tblgen_control_qpu = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_target_qpu;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'target_qpu'");
    if (namedAttrIt->getName() == getTargetQpuAttrName()) {
      tblgen_target_qpu = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_DQCDialect0(*this, tblgen_control_qpu, "control_qpu")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_DQCDialect0(*this, tblgen_target_qpu, "target_qpu")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TeleGateOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TeleGateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand control_qubitRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> control_qubitOperands(control_qubitRawOperands);  ::llvm::SMLoc control_qubitOperandsLoc;
  (void)control_qubitOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand target_qubitRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> target_qubitOperands(target_qubitRawOperands);  ::llvm::SMLoc target_qubitOperandsLoc;
  (void)target_qubitOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand epr_handleRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> epr_handleOperands(epr_handleRawOperands);  ::llvm::SMLoc epr_handleOperandsLoc;
  (void)epr_handleOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allOperandTypes;
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  control_qubitOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(control_qubitRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  target_qubitOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(target_qubitRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  epr_handleOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(epr_handleRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(::llvm::concat<const ::mlir::OpAsmParser::UnresolvedOperand>(control_qubitOperands, target_qubitOperands, epr_handleOperands), allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TeleGateOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getControlQubit();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getTargetQubit();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getEprHandle();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperandTypes();
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

} // namespace dqc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::dqc::TeleGateOp)


#endif  // GET_OP_CLASSES


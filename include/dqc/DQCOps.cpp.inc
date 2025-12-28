/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: DQCDialect.td                                                        *|
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

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_DQCDialect1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((llvm::isa<dqc::EPRHandleType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be epr_handle, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_DQCDialect2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((llvm::isa<dqc::QubitType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be qubit, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_DQCDialect3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((llvm::isa<dqc::QubitType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of qubit, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_DQCDialect1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_DQCDialect1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_DQCDialect1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_DQCDialect2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::FloatAttr>(attr))) && ((::llvm::cast<::mlir::FloatAttr>(attr).getType().isF32()))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit float attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_DQCDialect2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_DQCDialect2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_DQCDialect3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::DictionaryAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: dictionary of named attribute values";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_DQCDialect3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_DQCDialect3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_DQCDialect4(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))); }))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit integer array attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_DQCDialect4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_DQCDialect4(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
} // namespace dqc
namespace dqc {

//===----------------------------------------------------------------------===//
// ::dqc::EPRAllocOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
::mlir::IntegerAttr EPRAllocOpGenericAdaptorBase::getSourceQpuAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, EPRAllocOp::getSourceQpuAttrName(*odsOpName)));
  return attr;
}

uint32_t EPRAllocOpGenericAdaptorBase::getSourceQpu() {
  auto attr = getSourceQpuAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr EPRAllocOpGenericAdaptorBase::getTargetQpuAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, EPRAllocOp::getTargetQpuAttrName(*odsOpName)));
  return attr;
}

uint32_t EPRAllocOpGenericAdaptorBase::getTargetQpu() {
  auto attr = getTargetQpuAttr();
  return attr.getValue().getZExtValue();
}

::mlir::FloatAttr EPRAllocOpGenericAdaptorBase::getFidelityAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, EPRAllocOp::getFidelityAttrName(*odsOpName)));
  return attr;
}

::std::optional< ::llvm::APFloat > EPRAllocOpGenericAdaptorBase::getFidelity() {
  auto attr = getFidelityAttr();
  return attr ? ::std::optional< ::llvm::APFloat >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
EPRAllocOpAdaptor::EPRAllocOpAdaptor(EPRAllocOp op) : EPRAllocOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult EPRAllocOpAdaptor::verify(::mlir::Location loc) {
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

  if (tblgen_source_qpu && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_source_qpu))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_source_qpu).getType().isSignlessInteger(32)))))
    return emitError(loc, "'dqc.epr_alloc' op ""attribute 'source_qpu' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_target_qpu && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_target_qpu))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_target_qpu).getType().isSignlessInteger(32)))))
    return emitError(loc, "'dqc.epr_alloc' op ""attribute 'target_qpu' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_fidelity && !(((::llvm::isa<::mlir::FloatAttr>(tblgen_fidelity))) && ((::llvm::cast<::mlir::FloatAttr>(tblgen_fidelity).getType().isF32()))))
    return emitError(loc, "'dqc.epr_alloc' op ""attribute 'fidelity' failed to satisfy constraint: 32-bit float attribute");
  return ::mlir::success();
}

uint32_t EPRAllocOp::getSourceQpu() {
  auto attr = getSourceQpuAttr();
  return attr.getValue().getZExtValue();
}

uint32_t EPRAllocOp::getTargetQpu() {
  auto attr = getTargetQpuAttr();
  return attr.getValue().getZExtValue();
}

::std::optional< ::llvm::APFloat > EPRAllocOp::getFidelity() {
  auto attr = getFidelityAttr();
  return attr ? ::std::optional< ::llvm::APFloat >(attr.getValue()) : (::std::nullopt);
}

void EPRAllocOp::setSourceQpu(uint32_t attrValue) {
  (*this)->setAttr(getSourceQpuAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void EPRAllocOp::setTargetQpu(uint32_t attrValue) {
  (*this)->setAttr(getTargetQpuAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void EPRAllocOp::setFidelity(::std::optional<::llvm::APFloat> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getFidelityAttrName(), ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), *attrValue));
    (*this)->removeAttr(getFidelityAttrName());
}

void EPRAllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type epr_handle, ::mlir::IntegerAttr source_qpu, ::mlir::IntegerAttr target_qpu, /*optional*/::mlir::FloatAttr fidelity) {
  odsState.addAttribute(getSourceQpuAttrName(odsState.name), source_qpu);
  odsState.addAttribute(getTargetQpuAttrName(odsState.name), target_qpu);
  if (fidelity) {
    odsState.addAttribute(getFidelityAttrName(odsState.name), fidelity);
  }
  odsState.addTypes(epr_handle);
}

EPRAllocOp EPRAllocOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type epr_handle, ::mlir::IntegerAttr source_qpu, ::mlir::IntegerAttr target_qpu, /*optional*/::mlir::FloatAttr fidelity) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, epr_handle, source_qpu, target_qpu, fidelity);
  auto __res__ = ::llvm::dyn_cast<EPRAllocOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

EPRAllocOp EPRAllocOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type epr_handle, ::mlir::IntegerAttr source_qpu, ::mlir::IntegerAttr target_qpu, /*optional*/::mlir::FloatAttr fidelity) {
  return create(builder, builder.getLoc(), epr_handle, source_qpu, target_qpu, fidelity);
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

EPRAllocOp EPRAllocOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr source_qpu, ::mlir::IntegerAttr target_qpu, /*optional*/::mlir::FloatAttr fidelity) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, source_qpu, target_qpu, fidelity);
  auto __res__ = ::llvm::dyn_cast<EPRAllocOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

EPRAllocOp EPRAllocOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr source_qpu, ::mlir::IntegerAttr target_qpu, /*optional*/::mlir::FloatAttr fidelity) {
  return create(builder, builder.getLoc(), resultTypes, source_qpu, target_qpu, fidelity);
}

void EPRAllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type epr_handle, uint32_t source_qpu, uint32_t target_qpu, /*optional*/::mlir::FloatAttr fidelity) {
  odsState.addAttribute(getSourceQpuAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), source_qpu));
  odsState.addAttribute(getTargetQpuAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), target_qpu));
  if (fidelity) {
    odsState.addAttribute(getFidelityAttrName(odsState.name), fidelity);
  }
  odsState.addTypes(epr_handle);
}

EPRAllocOp EPRAllocOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type epr_handle, uint32_t source_qpu, uint32_t target_qpu, /*optional*/::mlir::FloatAttr fidelity) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, epr_handle, source_qpu, target_qpu, fidelity);
  auto __res__ = ::llvm::dyn_cast<EPRAllocOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

EPRAllocOp EPRAllocOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type epr_handle, uint32_t source_qpu, uint32_t target_qpu, /*optional*/::mlir::FloatAttr fidelity) {
  return create(builder, builder.getLoc(), epr_handle, source_qpu, target_qpu, fidelity);
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

EPRAllocOp EPRAllocOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint32_t source_qpu, uint32_t target_qpu, /*optional*/::mlir::FloatAttr fidelity) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, source_qpu, target_qpu, fidelity);
  auto __res__ = ::llvm::dyn_cast<EPRAllocOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

EPRAllocOp EPRAllocOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint32_t source_qpu, uint32_t target_qpu, /*optional*/::mlir::FloatAttr fidelity) {
  return create(builder, builder.getLoc(), resultTypes, source_qpu, target_qpu, fidelity);
}

void EPRAllocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

EPRAllocOp EPRAllocOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, operands, attributes);
  auto __res__ = ::llvm::dyn_cast<EPRAllocOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

EPRAllocOp EPRAllocOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), resultTypes, operands, attributes);
}

::llvm::LogicalResult EPRAllocOp::verifyInvariantsImpl() {
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_DQCDialect1(*this, tblgen_source_qpu, "source_qpu")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_DQCDialect1(*this, tblgen_target_qpu, "target_qpu")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_DQCDialect2(*this, tblgen_fidelity, "fidelity")))
    return ::mlir::failure();
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

::llvm::LogicalResult EPRAllocOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult EPRAllocOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr source_qpuAttr;
  ::mlir::IntegerAttr target_qpuAttr;
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  if (parser.parseCustomAttributeWithFallback(source_qpuAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (source_qpuAttr) result.attributes.append("source_qpu", source_qpuAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(target_qpuAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (target_qpuAttr) result.attributes.append("target_qpu", target_qpuAttr);
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
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
} // namespace detail
EPRConsumeOpAdaptor::EPRConsumeOpAdaptor(EPRConsumeOp op) : EPRConsumeOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult EPRConsumeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void EPRConsumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value epr_handle) {
  odsState.addOperands(epr_handle);
}

EPRConsumeOp EPRConsumeOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value epr_handle) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, epr_handle);
  auto __res__ = ::llvm::dyn_cast<EPRConsumeOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

EPRConsumeOp EPRConsumeOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value epr_handle) {
  return create(builder, builder.getLoc(), epr_handle);
}

void EPRConsumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value epr_handle) {
  odsState.addOperands(epr_handle);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

EPRConsumeOp EPRConsumeOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value epr_handle) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, epr_handle);
  auto __res__ = ::llvm::dyn_cast<EPRConsumeOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

EPRConsumeOp EPRConsumeOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value epr_handle) {
  return create(builder, builder.getLoc(), resultTypes, epr_handle);
}

void EPRConsumeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

EPRConsumeOp EPRConsumeOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, operands, attributes);
  auto __res__ = ::llvm::dyn_cast<EPRConsumeOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

EPRConsumeOp EPRConsumeOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), resultTypes, operands, attributes);
}

::llvm::LogicalResult EPRConsumeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult EPRConsumeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult EPRConsumeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand epr_handleRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> epr_handleOperands(&epr_handleRawOperand, 1);  ::llvm::SMLoc epr_handleOperandsLoc;
  (void)epr_handleOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allOperandTypes;

  epr_handleOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(epr_handleRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
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
::mlir::DictionaryAttr PartitionInfoOpGenericAdaptorBase::getPartitionMapAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PartitionInfoOp::getPartitionMapAttrName(*odsOpName)));
  return attr;
}

::mlir::DictionaryAttr PartitionInfoOpGenericAdaptorBase::getPartitionMap() {
  auto attr = getPartitionMapAttr();
  return attr;
}

} // namespace detail
PartitionInfoOpAdaptor::PartitionInfoOpAdaptor(PartitionInfoOp op) : PartitionInfoOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult PartitionInfoOpAdaptor::verify(::mlir::Location loc) {
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

  if (tblgen_partition_map && !((::llvm::isa<::mlir::DictionaryAttr>(tblgen_partition_map))))
    return emitError(loc, "'dqc.partition_info' op ""attribute 'partition_map' failed to satisfy constraint: dictionary of named attribute values");
  return ::mlir::success();
}

::mlir::DictionaryAttr PartitionInfoOp::getPartitionMap() {
  auto attr = getPartitionMapAttr();
  return attr;
}

void PartitionInfoOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::DictionaryAttr partition_map) {
  odsState.addAttribute(getPartitionMapAttrName(odsState.name), partition_map);
}

PartitionInfoOp PartitionInfoOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::DictionaryAttr partition_map) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, partition_map);
  auto __res__ = ::llvm::dyn_cast<PartitionInfoOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

PartitionInfoOp PartitionInfoOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::DictionaryAttr partition_map) {
  return create(builder, builder.getLoc(), partition_map);
}

void PartitionInfoOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DictionaryAttr partition_map) {
  odsState.addAttribute(getPartitionMapAttrName(odsState.name), partition_map);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

PartitionInfoOp PartitionInfoOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::DictionaryAttr partition_map) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, partition_map);
  auto __res__ = ::llvm::dyn_cast<PartitionInfoOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

PartitionInfoOp PartitionInfoOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::DictionaryAttr partition_map) {
  return create(builder, builder.getLoc(), resultTypes, partition_map);
}

void PartitionInfoOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

PartitionInfoOp PartitionInfoOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, operands, attributes);
  auto __res__ = ::llvm::dyn_cast<PartitionInfoOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

PartitionInfoOp PartitionInfoOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), resultTypes, operands, attributes);
}

::llvm::LogicalResult PartitionInfoOp::verifyInvariantsImpl() {
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_DQCDialect3(*this, tblgen_partition_map, "partition_map")))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::LogicalResult PartitionInfoOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PartitionInfoOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
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

::mlir::IntegerAttr TeleGateMultiOpGenericAdaptorBase::getControlQpuAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TeleGateMultiOp::getControlQpuAttrName(*odsOpName)));
  return attr;
}

uint32_t TeleGateMultiOpGenericAdaptorBase::getControlQpu() {
  auto attr = getControlQpuAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr TeleGateMultiOpGenericAdaptorBase::getTargetQpusAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, TeleGateMultiOp::getTargetQpusAttrName(*odsOpName)));
  return attr;
}

::mlir::ArrayAttr TeleGateMultiOpGenericAdaptorBase::getTargetQpus() {
  auto attr = getTargetQpusAttr();
  return attr;
}

} // namespace detail
TeleGateMultiOpAdaptor::TeleGateMultiOpAdaptor(TeleGateMultiOp op) : TeleGateMultiOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TeleGateMultiOpAdaptor::verify(::mlir::Location loc) {
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

  if (tblgen_control_qpu && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_control_qpu))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_control_qpu).getType().isSignlessInteger(32)))))
    return emitError(loc, "'dqc.telegate_multi' op ""attribute 'control_qpu' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_target_qpus && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_target_qpus))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_target_qpus), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))); }))))
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

::mlir::MutableOperandRange TeleGateMultiOp::getTargetQubitsMutable() {
  auto range = getODSOperandIndexAndLength(1);
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

uint32_t TeleGateMultiOp::getControlQpu() {
  auto attr = getControlQpuAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr TeleGateMultiOp::getTargetQpus() {
  auto attr = getTargetQpusAttr();
  return attr;
}

void TeleGateMultiOp::setControlQpu(uint32_t attrValue) {
  (*this)->setAttr(getControlQpuAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void TeleGateMultiOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::Value control_qubit, ::mlir::ValueRange target_qubits, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::ArrayAttr target_qpus) {
  odsState.addOperands(control_qubit);
  odsState.addOperands(target_qubits);
  odsState.addOperands(epr_handle);
  odsState.addAttribute(getControlQpuAttrName(odsState.name), control_qpu);
  odsState.addAttribute(getTargetQpusAttrName(odsState.name), target_qpus);
  odsState.addTypes(results);
}

TeleGateMultiOp TeleGateMultiOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange results, ::mlir::Value control_qubit, ::mlir::ValueRange target_qubits, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::ArrayAttr target_qpus) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, results, control_qubit, target_qubits, epr_handle, control_qpu, target_qpus);
  auto __res__ = ::llvm::dyn_cast<TeleGateMultiOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TeleGateMultiOp TeleGateMultiOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange results, ::mlir::Value control_qubit, ::mlir::ValueRange target_qubits, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::ArrayAttr target_qpus) {
  return create(builder, builder.getLoc(), results, control_qubit, target_qubits, epr_handle, control_qpu, target_qpus);
}

void TeleGateMultiOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::Value control_qubit, ::mlir::ValueRange target_qubits, ::mlir::Value epr_handle, uint32_t control_qpu, ::mlir::ArrayAttr target_qpus) {
  odsState.addOperands(control_qubit);
  odsState.addOperands(target_qubits);
  odsState.addOperands(epr_handle);
  odsState.addAttribute(getControlQpuAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), control_qpu));
  odsState.addAttribute(getTargetQpusAttrName(odsState.name), target_qpus);
  odsState.addTypes(results);
}

TeleGateMultiOp TeleGateMultiOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange results, ::mlir::Value control_qubit, ::mlir::ValueRange target_qubits, ::mlir::Value epr_handle, uint32_t control_qpu, ::mlir::ArrayAttr target_qpus) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, results, control_qubit, target_qubits, epr_handle, control_qpu, target_qpus);
  auto __res__ = ::llvm::dyn_cast<TeleGateMultiOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TeleGateMultiOp TeleGateMultiOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange results, ::mlir::Value control_qubit, ::mlir::ValueRange target_qubits, ::mlir::Value epr_handle, uint32_t control_qpu, ::mlir::ArrayAttr target_qpus) {
  return create(builder, builder.getLoc(), results, control_qubit, target_qubits, epr_handle, control_qpu, target_qpus);
}

void TeleGateMultiOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

TeleGateMultiOp TeleGateMultiOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, operands, attributes);
  auto __res__ = ::llvm::dyn_cast<TeleGateMultiOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TeleGateMultiOp TeleGateMultiOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), resultTypes, operands, attributes);
}

::llvm::LogicalResult TeleGateMultiOp::verifyInvariantsImpl() {
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_DQCDialect1(*this, tblgen_control_qpu, "control_qpu")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_DQCDialect4(*this, tblgen_target_qpus, "target_qpus")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TeleGateMultiOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace dqc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::dqc::TeleGateMultiOp)

namespace dqc {

//===----------------------------------------------------------------------===//
// ::dqc::TeleGateOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
::mlir::IntegerAttr TeleGateOpGenericAdaptorBase::getControlQpuAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TeleGateOp::getControlQpuAttrName(*odsOpName)));
  return attr;
}

uint32_t TeleGateOpGenericAdaptorBase::getControlQpu() {
  auto attr = getControlQpuAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr TeleGateOpGenericAdaptorBase::getTargetQpuAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, TeleGateOp::getTargetQpuAttrName(*odsOpName)));
  return attr;
}

uint32_t TeleGateOpGenericAdaptorBase::getTargetQpu() {
  auto attr = getTargetQpuAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
TeleGateOpAdaptor::TeleGateOpAdaptor(TeleGateOp op) : TeleGateOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TeleGateOpAdaptor::verify(::mlir::Location loc) {
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

  if (tblgen_control_qpu && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_control_qpu))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_control_qpu).getType().isSignlessInteger(32)))))
    return emitError(loc, "'dqc.telegate' op ""attribute 'control_qpu' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_target_qpu && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_target_qpu))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_target_qpu).getType().isSignlessInteger(32)))))
    return emitError(loc, "'dqc.telegate' op ""attribute 'target_qpu' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

uint32_t TeleGateOp::getControlQpu() {
  auto attr = getControlQpuAttr();
  return attr.getValue().getZExtValue();
}

uint32_t TeleGateOp::getTargetQpu() {
  auto attr = getTargetQpuAttr();
  return attr.getValue().getZExtValue();
}

void TeleGateOp::setControlQpu(uint32_t attrValue) {
  (*this)->setAttr(getControlQpuAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
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

TeleGateOp TeleGateOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, result, control_qubit, target_qubit, epr_handle, control_qpu, target_qpu);
  auto __res__ = ::llvm::dyn_cast<TeleGateOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TeleGateOp TeleGateOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu) {
  return create(builder, builder.getLoc(), result, control_qubit, target_qubit, epr_handle, control_qpu, target_qpu);
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

TeleGateOp TeleGateOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, control_qubit, target_qubit, epr_handle, control_qpu, target_qpu);
  auto __res__ = ::llvm::dyn_cast<TeleGateOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TeleGateOp TeleGateOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu) {
  return create(builder, builder.getLoc(), resultTypes, control_qubit, target_qubit, epr_handle, control_qpu, target_qpu);
}

void TeleGateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu) {
  odsState.addOperands(control_qubit);
  odsState.addOperands(target_qubit);
  odsState.addOperands(epr_handle);
  odsState.addAttribute(getControlQpuAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), control_qpu));
  odsState.addAttribute(getTargetQpuAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), target_qpu));
  odsState.addTypes(result);
}

TeleGateOp TeleGateOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, result, control_qubit, target_qubit, epr_handle, control_qpu, target_qpu);
  auto __res__ = ::llvm::dyn_cast<TeleGateOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TeleGateOp TeleGateOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu) {
  return create(builder, builder.getLoc(), result, control_qubit, target_qubit, epr_handle, control_qpu, target_qpu);
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

TeleGateOp TeleGateOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, control_qubit, target_qubit, epr_handle, control_qpu, target_qpu);
  auto __res__ = ::llvm::dyn_cast<TeleGateOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TeleGateOp TeleGateOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu) {
  return create(builder, builder.getLoc(), resultTypes, control_qubit, target_qubit, epr_handle, control_qpu, target_qpu);
}

void TeleGateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

TeleGateOp TeleGateOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, operands, attributes);
  auto __res__ = ::llvm::dyn_cast<TeleGateOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TeleGateOp TeleGateOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), resultTypes, operands, attributes);
}

::llvm::LogicalResult TeleGateOp::verifyInvariantsImpl() {
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_DQCDialect1(*this, tblgen_control_qpu, "control_qpu")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_DQCDialect1(*this, tblgen_target_qpu, "target_qpu")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_DQCDialect2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TeleGateOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TeleGateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand control_qubitRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> control_qubitOperands(&control_qubitRawOperand, 1);  ::llvm::SMLoc control_qubitOperandsLoc;
  (void)control_qubitOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand target_qubitRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> target_qubitOperands(&target_qubitRawOperand, 1);  ::llvm::SMLoc target_qubitOperandsLoc;
  (void)target_qubitOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand epr_handleRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> epr_handleOperands(&epr_handleRawOperand, 1);  ::llvm::SMLoc epr_handleOperandsLoc;
  (void)epr_handleOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allOperandTypes;
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  control_qubitOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(control_qubitRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  target_qubitOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(target_qubitRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  epr_handleOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(epr_handleRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
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


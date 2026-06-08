/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: MPIDialect.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::dqc::mpi::DistributeEprOp,
::dqc::mpi::TeleGateSequenceOp
#endif // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//


static ::llvm::LogicalResult __mlir_ods_local_type_constraint_MPIDialect1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::dqc::isEprHandleType(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be EPR handle type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_MPIDialect2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::dqc::isQubitType(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be qubit type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_MPIDialect1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_MPIDialect1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_MPIDialect1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
namespace dqc::mpi {


//===----------------------------------------------------------------------===//
// ::dqc::mpi::DistributeEprOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

DistributeEprOpGenericAdaptorBase::DistributeEprOpGenericAdaptorBase(DistributeEprOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint32_t DistributeEprOpGenericAdaptorBase::getSrcQpu() {
  auto attr = getSrcQpuAttr();
  return attr.getValue().getZExtValue();
}

uint32_t DistributeEprOpGenericAdaptorBase::getTgtQpu() {
  auto attr = getTgtQpuAttr();
  return attr.getValue().getZExtValue();
}


} // namespace detail
DistributeEprOpAdaptor::DistributeEprOpAdaptor(DistributeEprOp op) : DistributeEprOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult DistributeEprOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_src_qpu = getProperties().src_qpu; (void)tblgen_src_qpu;
  if (!tblgen_src_qpu) return emitError(loc, "'mpi.distribute_epr' op requires attribute 'src_qpu'");
  auto tblgen_tgt_qpu = getProperties().tgt_qpu; (void)tblgen_tgt_qpu;
  if (!tblgen_tgt_qpu) return emitError(loc, "'mpi.distribute_epr' op requires attribute 'tgt_qpu'");

  if (tblgen_src_qpu && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_src_qpu))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_src_qpu).getType().isSignlessInteger(32)))))
    return emitError(loc, "'mpi.distribute_epr' op attribute 'src_qpu' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_tgt_qpu && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_tgt_qpu))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_tgt_qpu).getType().isSignlessInteger(32)))))
    return emitError(loc, "'mpi.distribute_epr' op attribute 'tgt_qpu' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult DistributeEprOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.src_qpu;
       auto attr = dict.get("src_qpu");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `src_qpu` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.tgt_qpu;
       auto attr = dict.get("tgt_qpu");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `tgt_qpu` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute DistributeEprOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.src_qpu;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("src_qpu",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.tgt_qpu;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("tgt_qpu",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code DistributeEprOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.src_qpu.getAsOpaquePointer()), 
    llvm::hash_value(prop.tgt_qpu.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> DistributeEprOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "src_qpu")
      return prop.src_qpu;

    if (name == "tgt_qpu")
      return prop.tgt_qpu;
  return std::nullopt;
}

void DistributeEprOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "src_qpu") {
       prop.src_qpu = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.src_qpu)>>(value);
       return;
    }

    if (name == "tgt_qpu") {
       prop.tgt_qpu = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.tgt_qpu)>>(value);
       return;
    }
}

void DistributeEprOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.src_qpu) attrs.append("src_qpu", prop.src_qpu);

    if (prop.tgt_qpu) attrs.append("tgt_qpu", prop.tgt_qpu);
}

::llvm::LogicalResult DistributeEprOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getSrcQpuAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_MPIDialect1(attr, "src_qpu", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getTgtQpuAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_MPIDialect1(attr, "tgt_qpu", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult DistributeEprOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.src_qpu)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.tgt_qpu)))
    return ::mlir::failure();
  return ::mlir::success();
}

void DistributeEprOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.src_qpu);
  writer.writeAttribute(prop.tgt_qpu);
}

uint32_t DistributeEprOp::getSrcQpu() {
  auto attr = getSrcQpuAttr();
  return attr.getValue().getZExtValue();
}

uint32_t DistributeEprOp::getTgtQpu() {
  auto attr = getTgtQpuAttr();
  return attr.getValue().getZExtValue();
}

void DistributeEprOp::setSrcQpu(uint32_t attrValue) {
  getProperties().src_qpu = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void DistributeEprOp::setTgtQpu(uint32_t attrValue) {
  getProperties().tgt_qpu = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void DistributeEprOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type epr_handle, ::mlir::IntegerAttr src_qpu, ::mlir::IntegerAttr tgt_qpu) {
  odsState.getOrAddProperties<Properties>().src_qpu = src_qpu;
  odsState.getOrAddProperties<Properties>().tgt_qpu = tgt_qpu;
  odsState.addTypes(epr_handle);
}

DistributeEprOp DistributeEprOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type epr_handle, ::mlir::IntegerAttr src_qpu, ::mlir::IntegerAttr tgt_qpu) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(epr_handle)>(epr_handle), std::forward<decltype(src_qpu)>(src_qpu), std::forward<decltype(tgt_qpu)>(tgt_qpu));
  auto __res__ = ::llvm::dyn_cast<DistributeEprOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

DistributeEprOp DistributeEprOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type epr_handle, ::mlir::IntegerAttr src_qpu, ::mlir::IntegerAttr tgt_qpu) {
  return create(builder, builder.getLoc(), std::forward<decltype(epr_handle)>(epr_handle), std::forward<decltype(src_qpu)>(src_qpu), std::forward<decltype(tgt_qpu)>(tgt_qpu));
}

void DistributeEprOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr src_qpu, ::mlir::IntegerAttr tgt_qpu) {
  odsState.getOrAddProperties<Properties>().src_qpu = src_qpu;
  odsState.getOrAddProperties<Properties>().tgt_qpu = tgt_qpu;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

DistributeEprOp DistributeEprOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr src_qpu, ::mlir::IntegerAttr tgt_qpu) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(src_qpu)>(src_qpu), std::forward<decltype(tgt_qpu)>(tgt_qpu));
  auto __res__ = ::llvm::dyn_cast<DistributeEprOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

DistributeEprOp DistributeEprOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr src_qpu, ::mlir::IntegerAttr tgt_qpu) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(src_qpu)>(src_qpu), std::forward<decltype(tgt_qpu)>(tgt_qpu));
}

void DistributeEprOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type epr_handle, uint32_t src_qpu, uint32_t tgt_qpu) {
  odsState.getOrAddProperties<Properties>().src_qpu = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), src_qpu);
  odsState.getOrAddProperties<Properties>().tgt_qpu = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), tgt_qpu);
  odsState.addTypes(epr_handle);
}

DistributeEprOp DistributeEprOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type epr_handle, uint32_t src_qpu, uint32_t tgt_qpu) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(epr_handle)>(epr_handle), std::forward<decltype(src_qpu)>(src_qpu), std::forward<decltype(tgt_qpu)>(tgt_qpu));
  auto __res__ = ::llvm::dyn_cast<DistributeEprOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

DistributeEprOp DistributeEprOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type epr_handle, uint32_t src_qpu, uint32_t tgt_qpu) {
  return create(builder, builder.getLoc(), std::forward<decltype(epr_handle)>(epr_handle), std::forward<decltype(src_qpu)>(src_qpu), std::forward<decltype(tgt_qpu)>(tgt_qpu));
}

void DistributeEprOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t src_qpu, uint32_t tgt_qpu) {
  odsState.getOrAddProperties<Properties>().src_qpu = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), src_qpu);
  odsState.getOrAddProperties<Properties>().tgt_qpu = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), tgt_qpu);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

DistributeEprOp DistributeEprOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint32_t src_qpu, uint32_t tgt_qpu) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(src_qpu)>(src_qpu), std::forward<decltype(tgt_qpu)>(tgt_qpu));
  auto __res__ = ::llvm::dyn_cast<DistributeEprOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

DistributeEprOp DistributeEprOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint32_t src_qpu, uint32_t tgt_qpu) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(src_qpu)>(src_qpu), std::forward<decltype(tgt_qpu)>(tgt_qpu));
}

void DistributeEprOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<DistributeEprOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

DistributeEprOp DistributeEprOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<DistributeEprOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

DistributeEprOp DistributeEprOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void DistributeEprOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

DistributeEprOp DistributeEprOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<DistributeEprOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

DistributeEprOp DistributeEprOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult DistributeEprOp::verifyInvariantsImpl() {
  auto tblgen_src_qpu = getProperties().src_qpu; (void)tblgen_src_qpu;
  if (!tblgen_src_qpu) return emitOpError("requires attribute 'src_qpu'");
  auto tblgen_tgt_qpu = getProperties().tgt_qpu; (void)tblgen_tgt_qpu;
  if (!tblgen_tgt_qpu) return emitOpError("requires attribute 'tgt_qpu'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MPIDialect1(*this, tblgen_src_qpu, "src_qpu")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MPIDialect1(*this, tblgen_tgt_qpu, "tgt_qpu")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MPIDialect1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult DistributeEprOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult DistributeEprOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr src_qpuAttr;
  ::mlir::IntegerAttr tgt_qpuAttr;
  ::mlir::Type epr_handleRawType{};
  ::llvm::ArrayRef<::mlir::Type> epr_handleTypes(&epr_handleRawType, 1);

  if (parser.parseCustomAttributeWithFallback(src_qpuAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (src_qpuAttr) result.getOrAddProperties<DistributeEprOp::Properties>().src_qpu = src_qpuAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(tgt_qpuAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (tgt_qpuAttr) result.getOrAddProperties<DistributeEprOp::Properties>().tgt_qpu = tgt_qpuAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    epr_handleRawType = type;
  }
  result.addTypes(epr_handleTypes);
  return ::mlir::success();
}

void DistributeEprOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getSrcQpuAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getTgtQpuAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("src_qpu");
  elidedAttrs.push_back("tgt_qpu");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getEprHandle().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}


} // namespace dqc::mpi
MLIR_DEFINE_EXPLICIT_TYPE_ID(::dqc::mpi::DistributeEprOp)

namespace dqc::mpi {


//===----------------------------------------------------------------------===//
// ::dqc::mpi::TeleGateSequenceOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

TeleGateSequenceOpGenericAdaptorBase::TeleGateSequenceOpGenericAdaptorBase(TeleGateSequenceOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint32_t TeleGateSequenceOpGenericAdaptorBase::getControlQpu() {
  auto attr = getControlQpuAttr();
  return attr.getValue().getZExtValue();
}

uint32_t TeleGateSequenceOpGenericAdaptorBase::getTargetQpu() {
  auto attr = getTargetQpuAttr();
  return attr.getValue().getZExtValue();
}


} // namespace detail
TeleGateSequenceOpAdaptor::TeleGateSequenceOpAdaptor(TeleGateSequenceOp op) : TeleGateSequenceOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TeleGateSequenceOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_control_qpu = getProperties().control_qpu; (void)tblgen_control_qpu;
  if (!tblgen_control_qpu) return emitError(loc, "'mpi.telegate_sequence' op requires attribute 'control_qpu'");
  auto tblgen_target_qpu = getProperties().target_qpu; (void)tblgen_target_qpu;
  if (!tblgen_target_qpu) return emitError(loc, "'mpi.telegate_sequence' op requires attribute 'target_qpu'");

  if (tblgen_control_qpu && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_control_qpu))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_control_qpu).getType().isSignlessInteger(32)))))
    return emitError(loc, "'mpi.telegate_sequence' op attribute 'control_qpu' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_target_qpu && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_target_qpu))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_target_qpu).getType().isSignlessInteger(32)))))
    return emitError(loc, "'mpi.telegate_sequence' op attribute 'target_qpu' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult TeleGateSequenceOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.control_qpu;
       auto attr = dict.get("control_qpu");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `control_qpu` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.target_qpu;
       auto attr = dict.get("target_qpu");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `target_qpu` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute TeleGateSequenceOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.control_qpu;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("control_qpu",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.target_qpu;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("target_qpu",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TeleGateSequenceOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.control_qpu.getAsOpaquePointer()), 
    llvm::hash_value(prop.target_qpu.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> TeleGateSequenceOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "control_qpu")
      return prop.control_qpu;

    if (name == "target_qpu")
      return prop.target_qpu;
  return std::nullopt;
}

void TeleGateSequenceOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "control_qpu") {
       prop.control_qpu = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.control_qpu)>>(value);
       return;
    }

    if (name == "target_qpu") {
       prop.target_qpu = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.target_qpu)>>(value);
       return;
    }
}

void TeleGateSequenceOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.control_qpu) attrs.append("control_qpu", prop.control_qpu);

    if (prop.target_qpu) attrs.append("target_qpu", prop.target_qpu);
}

::llvm::LogicalResult TeleGateSequenceOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getControlQpuAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_MPIDialect1(attr, "control_qpu", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getTargetQpuAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_MPIDialect1(attr, "target_qpu", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult TeleGateSequenceOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.control_qpu)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.target_qpu)))
    return ::mlir::failure();
  return ::mlir::success();
}

void TeleGateSequenceOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.control_qpu);
  writer.writeAttribute(prop.target_qpu);
}

uint32_t TeleGateSequenceOp::getControlQpu() {
  auto attr = getControlQpuAttr();
  return attr.getValue().getZExtValue();
}

uint32_t TeleGateSequenceOp::getTargetQpu() {
  auto attr = getTargetQpuAttr();
  return attr.getValue().getZExtValue();
}

void TeleGateSequenceOp::setControlQpu(uint32_t attrValue) {
  getProperties().control_qpu = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void TeleGateSequenceOp::setTargetQpu(uint32_t attrValue) {
  getProperties().target_qpu = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void TeleGateSequenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu) {
  odsState.addOperands(ctrl);
  odsState.addOperands(target);
  odsState.addOperands(epr_handle);
  odsState.getOrAddProperties<Properties>().control_qpu = control_qpu;
  odsState.getOrAddProperties<Properties>().target_qpu = target_qpu;
}

TeleGateSequenceOp TeleGateSequenceOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(ctrl)>(ctrl), std::forward<decltype(target)>(target), std::forward<decltype(epr_handle)>(epr_handle), std::forward<decltype(control_qpu)>(control_qpu), std::forward<decltype(target_qpu)>(target_qpu));
  auto __res__ = ::llvm::dyn_cast<TeleGateSequenceOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TeleGateSequenceOp TeleGateSequenceOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu) {
  return create(builder, builder.getLoc(), std::forward<decltype(ctrl)>(ctrl), std::forward<decltype(target)>(target), std::forward<decltype(epr_handle)>(epr_handle), std::forward<decltype(control_qpu)>(control_qpu), std::forward<decltype(target_qpu)>(target_qpu));
}

void TeleGateSequenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu) {
  odsState.addOperands(ctrl);
  odsState.addOperands(target);
  odsState.addOperands(epr_handle);
  odsState.getOrAddProperties<Properties>().control_qpu = control_qpu;
  odsState.getOrAddProperties<Properties>().target_qpu = target_qpu;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

TeleGateSequenceOp TeleGateSequenceOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(ctrl)>(ctrl), std::forward<decltype(target)>(target), std::forward<decltype(epr_handle)>(epr_handle), std::forward<decltype(control_qpu)>(control_qpu), std::forward<decltype(target_qpu)>(target_qpu));
  auto __res__ = ::llvm::dyn_cast<TeleGateSequenceOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TeleGateSequenceOp TeleGateSequenceOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(ctrl)>(ctrl), std::forward<decltype(target)>(target), std::forward<decltype(epr_handle)>(epr_handle), std::forward<decltype(control_qpu)>(control_qpu), std::forward<decltype(target_qpu)>(target_qpu));
}

void TeleGateSequenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu) {
  odsState.addOperands(ctrl);
  odsState.addOperands(target);
  odsState.addOperands(epr_handle);
  odsState.getOrAddProperties<Properties>().control_qpu = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), control_qpu);
  odsState.getOrAddProperties<Properties>().target_qpu = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), target_qpu);
}

TeleGateSequenceOp TeleGateSequenceOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(ctrl)>(ctrl), std::forward<decltype(target)>(target), std::forward<decltype(epr_handle)>(epr_handle), std::forward<decltype(control_qpu)>(control_qpu), std::forward<decltype(target_qpu)>(target_qpu));
  auto __res__ = ::llvm::dyn_cast<TeleGateSequenceOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TeleGateSequenceOp TeleGateSequenceOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu) {
  return create(builder, builder.getLoc(), std::forward<decltype(ctrl)>(ctrl), std::forward<decltype(target)>(target), std::forward<decltype(epr_handle)>(epr_handle), std::forward<decltype(control_qpu)>(control_qpu), std::forward<decltype(target_qpu)>(target_qpu));
}

void TeleGateSequenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu) {
  odsState.addOperands(ctrl);
  odsState.addOperands(target);
  odsState.addOperands(epr_handle);
  odsState.getOrAddProperties<Properties>().control_qpu = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), control_qpu);
  odsState.getOrAddProperties<Properties>().target_qpu = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), target_qpu);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

TeleGateSequenceOp TeleGateSequenceOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(ctrl)>(ctrl), std::forward<decltype(target)>(target), std::forward<decltype(epr_handle)>(epr_handle), std::forward<decltype(control_qpu)>(control_qpu), std::forward<decltype(target_qpu)>(target_qpu));
  auto __res__ = ::llvm::dyn_cast<TeleGateSequenceOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TeleGateSequenceOp TeleGateSequenceOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(ctrl)>(ctrl), std::forward<decltype(target)>(target), std::forward<decltype(epr_handle)>(epr_handle), std::forward<decltype(control_qpu)>(control_qpu), std::forward<decltype(target_qpu)>(target_qpu));
}

void TeleGateSequenceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<TeleGateSequenceOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

TeleGateSequenceOp TeleGateSequenceOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<TeleGateSequenceOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TeleGateSequenceOp TeleGateSequenceOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void TeleGateSequenceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

TeleGateSequenceOp TeleGateSequenceOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<TeleGateSequenceOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TeleGateSequenceOp TeleGateSequenceOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult TeleGateSequenceOp::verifyInvariantsImpl() {
  auto tblgen_control_qpu = getProperties().control_qpu; (void)tblgen_control_qpu;
  if (!tblgen_control_qpu) return emitOpError("requires attribute 'control_qpu'");
  auto tblgen_target_qpu = getProperties().target_qpu; (void)tblgen_target_qpu;
  if (!tblgen_target_qpu) return emitOpError("requires attribute 'target_qpu'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MPIDialect1(*this, tblgen_control_qpu, "control_qpu")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MPIDialect1(*this, tblgen_target_qpu, "target_qpu")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MPIDialect2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MPIDialect2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MPIDialect1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TeleGateSequenceOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TeleGateSequenceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand ctrlRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> ctrlOperands(&ctrlRawOperand, 1);  ::llvm::SMLoc ctrlOperandsLoc;
  (void)ctrlOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(&targetRawOperand, 1);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand epr_handleRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> epr_handleOperands(&epr_handleRawOperand, 1);  ::llvm::SMLoc epr_handleOperandsLoc;
  (void)epr_handleOperandsLoc;
  ::mlir::Type ctrlRawType{};
  ::llvm::ArrayRef<::mlir::Type> ctrlTypes(&ctrlRawType, 1);
  ::mlir::Type targetRawType{};
  ::llvm::ArrayRef<::mlir::Type> targetTypes(&targetRawType, 1);
  ::mlir::Type epr_handleRawType{};
  ::llvm::ArrayRef<::mlir::Type> epr_handleTypes(&epr_handleRawType, 1);

  ctrlOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(ctrlRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperand))
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
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    ctrlRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    targetRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    epr_handleRawType = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.resolveOperands(ctrlOperands, ctrlTypes, ctrlOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(targetOperands, targetTypes, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(epr_handleOperands, epr_handleTypes, epr_handleOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TeleGateSequenceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getCtrl();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getEprHandle();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getCtrl().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getTarget().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getEprHandle().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
}


} // namespace dqc::mpi
MLIR_DEFINE_EXPLICIT_TYPE_ID(::dqc::mpi::TeleGateSequenceOp)


#endif // GET_OP_CLASSES


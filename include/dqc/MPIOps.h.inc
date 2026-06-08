/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: MPIDialect.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace dqc::mpi {

/// Distribute an EPR pair between two QPUs
class DistributeEprOp;

} // namespace dqc::mpi
namespace dqc::mpi {

/// Execute a telegate sequence between two qubits via EPR
class TeleGateSequenceOp;

} // namespace dqc::mpi
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace dqc::mpi {


//===----------------------------------------------------------------------===//
// ::dqc::mpi::DistributeEprOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class DistributeEprOpGenericAdaptorBase {
public:
  struct Properties {
    using src_qpuTy = ::mlir::IntegerAttr;
    src_qpuTy src_qpu;

    auto getSrcQpu() const {
      auto &propStorage = this->src_qpu;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setSrcQpu(const ::mlir::IntegerAttr &propValue) {
      this->src_qpu = propValue;
    }
    using tgt_qpuTy = ::mlir::IntegerAttr;
    tgt_qpuTy tgt_qpu;

    auto getTgtQpu() const {
      auto &propStorage = this->tgt_qpu;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setTgtQpu(const ::mlir::IntegerAttr &propValue) {
      this->tgt_qpu = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.src_qpu == this->src_qpu &&
        rhs.tgt_qpu == this->tgt_qpu &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  DistributeEprOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.distribute_epr", odsAttrs.getContext());
  }

  DistributeEprOpGenericAdaptorBase(DistributeEprOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getSrcQpuAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().src_qpu);
    return attr;
  }

  uint32_t getSrcQpu();
  ::mlir::IntegerAttr getTgtQpuAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().tgt_qpu);
    return attr;
  }

  uint32_t getTgtQpu();
};

} // namespace detail
template <typename RangeT>
class DistributeEprOpGenericAdaptor : public detail::DistributeEprOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DistributeEprOpGenericAdaptorBase;
public:
  DistributeEprOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  DistributeEprOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : DistributeEprOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  DistributeEprOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : DistributeEprOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  DistributeEprOpGenericAdaptor(RangeT values, const DistributeEprOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = DistributeEprOp, typename = std::enable_if_t<std::is_same_v<LateInst, DistributeEprOp>>>
  DistributeEprOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DistributeEprOpAdaptor : public DistributeEprOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DistributeEprOpGenericAdaptor::DistributeEprOpGenericAdaptor;
  DistributeEprOpAdaptor(DistributeEprOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class DistributeEprOp : public ::mlir::Op<DistributeEprOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DistributeEprOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DistributeEprOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("src_qpu"), ::llvm::StringRef("tgt_qpu")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getSrcQpuAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getSrcQpuAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getTgtQpuAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getTgtQpuAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.distribute_epr");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getEprHandle() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getSrcQpuAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().src_qpu);
  }

  uint32_t getSrcQpu();
  ::mlir::IntegerAttr getTgtQpuAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().tgt_qpu);
  }

  uint32_t getTgtQpu();
  void setSrcQpuAttr(::mlir::IntegerAttr attr) {
    getProperties().src_qpu = attr;
  }

  void setSrcQpu(uint32_t attrValue);
  void setTgtQpuAttr(::mlir::IntegerAttr attr) {
    getProperties().tgt_qpu = attr;
  }

  void setTgtQpu(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type epr_handle, ::mlir::IntegerAttr src_qpu, ::mlir::IntegerAttr tgt_qpu);
  static DistributeEprOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type epr_handle, ::mlir::IntegerAttr src_qpu, ::mlir::IntegerAttr tgt_qpu);
  static DistributeEprOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type epr_handle, ::mlir::IntegerAttr src_qpu, ::mlir::IntegerAttr tgt_qpu);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr src_qpu, ::mlir::IntegerAttr tgt_qpu);
  static DistributeEprOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr src_qpu, ::mlir::IntegerAttr tgt_qpu);
  static DistributeEprOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr src_qpu, ::mlir::IntegerAttr tgt_qpu);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type epr_handle, uint32_t src_qpu, uint32_t tgt_qpu);
  static DistributeEprOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type epr_handle, uint32_t src_qpu, uint32_t tgt_qpu);
  static DistributeEprOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type epr_handle, uint32_t src_qpu, uint32_t tgt_qpu);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t src_qpu, uint32_t tgt_qpu);
  static DistributeEprOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint32_t src_qpu, uint32_t tgt_qpu);
  static DistributeEprOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint32_t src_qpu, uint32_t tgt_qpu);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static DistributeEprOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static DistributeEprOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static DistributeEprOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static DistributeEprOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace dqc::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::dqc::mpi::DistributeEprOp)

namespace dqc::mpi {


//===----------------------------------------------------------------------===//
// ::dqc::mpi::TeleGateSequenceOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class TeleGateSequenceOpGenericAdaptorBase {
public:
  struct Properties {
    using control_qpuTy = ::mlir::IntegerAttr;
    control_qpuTy control_qpu;

    auto getControlQpu() const {
      auto &propStorage = this->control_qpu;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setControlQpu(const ::mlir::IntegerAttr &propValue) {
      this->control_qpu = propValue;
    }
    using target_qpuTy = ::mlir::IntegerAttr;
    target_qpuTy target_qpu;

    auto getTargetQpu() const {
      auto &propStorage = this->target_qpu;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setTargetQpu(const ::mlir::IntegerAttr &propValue) {
      this->target_qpu = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.control_qpu == this->control_qpu &&
        rhs.target_qpu == this->target_qpu &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  TeleGateSequenceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.telegate_sequence", odsAttrs.getContext());
  }

  TeleGateSequenceOpGenericAdaptorBase(TeleGateSequenceOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getControlQpuAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().control_qpu);
    return attr;
  }

  uint32_t getControlQpu();
  ::mlir::IntegerAttr getTargetQpuAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().target_qpu);
    return attr;
  }

  uint32_t getTargetQpu();
};

} // namespace detail
template <typename RangeT>
class TeleGateSequenceOpGenericAdaptor : public detail::TeleGateSequenceOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TeleGateSequenceOpGenericAdaptorBase;
  static constexpr int odsIndex_ctrl = 0;
  static constexpr int odsIndex_target = 1;
  static constexpr int odsIndex_epr_handle = 2;
public:
  TeleGateSequenceOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TeleGateSequenceOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TeleGateSequenceOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  TeleGateSequenceOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : TeleGateSequenceOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  TeleGateSequenceOpGenericAdaptor(RangeT values, const TeleGateSequenceOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = TeleGateSequenceOp, typename = std::enable_if_t<std::is_same_v<LateInst, TeleGateSequenceOp>>>
  TeleGateSequenceOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getCtrl() {
    return (*getODSOperands(0).begin());
  }

  ValueT getTarget() {
    return (*getODSOperands(1).begin());
  }

  ValueT getEprHandle() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TeleGateSequenceOpAdaptor : public TeleGateSequenceOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TeleGateSequenceOpGenericAdaptor::TeleGateSequenceOpGenericAdaptor;
  TeleGateSequenceOpAdaptor(TeleGateSequenceOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TeleGateSequenceOp : public ::mlir::Op<TeleGateSequenceOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TeleGateSequenceOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TeleGateSequenceOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_ctrl = 0;
  static constexpr int odsIndex_target = 1;
  static constexpr int odsIndex_epr_handle = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("control_qpu"), ::llvm::StringRef("target_qpu")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getControlQpuAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getControlQpuAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getTargetQpuAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getTargetQpuAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.telegate_sequence");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getCtrl() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getTarget() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getEprHandle() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getCtrlMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getTargetMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getEprHandleMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getControlQpuAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().control_qpu);
  }

  uint32_t getControlQpu();
  ::mlir::IntegerAttr getTargetQpuAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().target_qpu);
  }

  uint32_t getTargetQpu();
  void setControlQpuAttr(::mlir::IntegerAttr attr) {
    getProperties().control_qpu = attr;
  }

  void setControlQpu(uint32_t attrValue);
  void setTargetQpuAttr(::mlir::IntegerAttr attr) {
    getProperties().target_qpu = attr;
  }

  void setTargetQpu(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu);
  static TeleGateSequenceOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu);
  static TeleGateSequenceOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu);
  static TeleGateSequenceOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu);
  static TeleGateSequenceOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu);
  static TeleGateSequenceOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu);
  static TeleGateSequenceOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu);
  static TeleGateSequenceOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu);
  static TeleGateSequenceOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ctrl, ::mlir::Value target, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TeleGateSequenceOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TeleGateSequenceOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TeleGateSequenceOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TeleGateSequenceOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace dqc::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::dqc::mpi::TeleGateSequenceOp)


#endif // GET_OP_CLASSES


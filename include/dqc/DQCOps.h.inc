/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: DQCDialect.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace dqc {
/// Allocate a shared entangled pair (Bell state) between two QPUs
/// Allocates a maximally entangled pair (Bell state) across two remote QPUs.
///     This is the fundamental resource for quantum teleportation-based gates.
///     
///     Example:
///     ```mlir
///     %epr = dqc.epr_alloc %qpu0, %qpu1 : i32, i32 -> !dqc.epr_handle
///     ```
class EPRAllocOp;
} // namespace dqc
namespace dqc {
/// Mark EPR pair consumption and enforce synchronization
/// Explicit operation denoting that an EPR pair has been consumed.
///     Useful for tracking resource usage and enforcing ordering constraints.
class EPRConsumeOp;
} // namespace dqc
namespace dqc {
/// Metadata operation storing qubit-to-QPU partition mapping
/// This operation stores the result of the hypergraph partitioning phase.
///     It maps logical qubit IDs to physical QPU IDs.
///     
///     Example:
///     ```mlir
///     dqc.partition_info {
///       qubit_0 = 0,
///       qubit_1 = 0,
///       qubit_2 = 1,
///       qubit_3 = 1
///     }
///     ```
class PartitionInfoOp;
} // namespace dqc
namespace dqc {
/// Perform multiple distributed gates sharing a single EPR pair
/// Optimized variant of telegate that applies multiple gates (gate packet)
///     that share a common control qubit using a single EPR pair allocation.
///     This reduces e-bit consumption by up to 30%.
class TeleGateMultiOp;
} // namespace dqc
namespace dqc {
/// Perform a distributed gate via quantum teleportation
/// Replaces a non-local two-qubit gate (primarily CNOT) with a teleportation
///     sequence. Requires a pre-allocated EPR pair between the control and target QPUs.
///     
///     Sequence:
///     1. Bell measurement on (control_qubit, epr_qubit_a)
///     2. Measurement outcomes sent as classical bits to target QPU
///     3. Target QPU applies corrective gate to (epr_qubit_b, target_qubit)
///     
///     Example:
///     ```mlir
///     %result = dqc.telegate %control, %target, %epr_handle 
///                           : !quir.qubit, !quir.qubit, !dqc.epr_handle
///                           -> !quir.qubit
///     ```
class TeleGateOp;
} // namespace dqc
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace dqc {

//===----------------------------------------------------------------------===//
// ::dqc::EPRAllocOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class EPRAllocOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  EPRAllocOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("dqc.epr_alloc", odsAttrs.getContext());
  }

  EPRAllocOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getSourceQpuAttr();
  uint32_t getSourceQpu();
  ::mlir::IntegerAttr getTargetQpuAttr();
  uint32_t getTargetQpu();
  ::mlir::FloatAttr getFidelityAttr();
  ::std::optional< ::llvm::APFloat > getFidelity();
};
} // namespace detail
template <typename RangeT>
class EPRAllocOpGenericAdaptor : public detail::EPRAllocOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::EPRAllocOpGenericAdaptorBase;
public:
  EPRAllocOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  EPRAllocOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : EPRAllocOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  EPRAllocOpGenericAdaptor(RangeT values, const EPRAllocOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = EPRAllocOp, typename = std::enable_if_t<std::is_same_v<LateInst, EPRAllocOp>>>
  EPRAllocOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

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
class EPRAllocOpAdaptor : public EPRAllocOpGenericAdaptor<::mlir::ValueRange> {
public:
  using EPRAllocOpGenericAdaptor::EPRAllocOpGenericAdaptor;
  EPRAllocOpAdaptor(EPRAllocOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class EPRAllocOp : public ::mlir::Op<EPRAllocOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = EPRAllocOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = EPRAllocOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("fidelity"), ::llvm::StringRef("source_qpu"), ::llvm::StringRef("target_qpu")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getFidelityAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getFidelityAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSourceQpuAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSourceQpuAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getTargetQpuAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getTargetQpuAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("dqc.epr_alloc");
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

  ::mlir::IntegerAttr getSourceQpuAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getSourceQpuAttrName()));
  }

  uint32_t getSourceQpu();
  ::mlir::IntegerAttr getTargetQpuAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTargetQpuAttrName()));
  }

  uint32_t getTargetQpu();
  ::mlir::FloatAttr getFidelityAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getFidelityAttrName()));
  }

  ::std::optional< ::llvm::APFloat > getFidelity();
  void setSourceQpuAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getSourceQpuAttrName(), attr);
  }

  void setSourceQpu(uint32_t attrValue);
  void setTargetQpuAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getTargetQpuAttrName(), attr);
  }

  void setTargetQpu(uint32_t attrValue);
  void setFidelityAttr(::mlir::FloatAttr attr) {
    (*this)->setAttr(getFidelityAttrName(), attr);
  }

  void setFidelity(::std::optional<::llvm::APFloat> attrValue);
  ::mlir::Attribute removeFidelityAttr() {
  return (*this)->removeAttr(getFidelityAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type epr_handle, ::mlir::IntegerAttr source_qpu, ::mlir::IntegerAttr target_qpu, /*optional*/::mlir::FloatAttr fidelity);
  static EPRAllocOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type epr_handle, ::mlir::IntegerAttr source_qpu, ::mlir::IntegerAttr target_qpu, /*optional*/::mlir::FloatAttr fidelity);
  static EPRAllocOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type epr_handle, ::mlir::IntegerAttr source_qpu, ::mlir::IntegerAttr target_qpu, /*optional*/::mlir::FloatAttr fidelity);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr source_qpu, ::mlir::IntegerAttr target_qpu, /*optional*/::mlir::FloatAttr fidelity);
  static EPRAllocOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr source_qpu, ::mlir::IntegerAttr target_qpu, /*optional*/::mlir::FloatAttr fidelity);
  static EPRAllocOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr source_qpu, ::mlir::IntegerAttr target_qpu, /*optional*/::mlir::FloatAttr fidelity);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type epr_handle, uint32_t source_qpu, uint32_t target_qpu, /*optional*/::mlir::FloatAttr fidelity);
  static EPRAllocOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type epr_handle, uint32_t source_qpu, uint32_t target_qpu, /*optional*/::mlir::FloatAttr fidelity);
  static EPRAllocOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type epr_handle, uint32_t source_qpu, uint32_t target_qpu, /*optional*/::mlir::FloatAttr fidelity);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t source_qpu, uint32_t target_qpu, /*optional*/::mlir::FloatAttr fidelity);
  static EPRAllocOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint32_t source_qpu, uint32_t target_qpu, /*optional*/::mlir::FloatAttr fidelity);
  static EPRAllocOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint32_t source_qpu, uint32_t target_qpu, /*optional*/::mlir::FloatAttr fidelity);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static EPRAllocOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static EPRAllocOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace dqc
MLIR_DECLARE_EXPLICIT_TYPE_ID(::dqc::EPRAllocOp)

namespace dqc {

//===----------------------------------------------------------------------===//
// ::dqc::EPRConsumeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class EPRConsumeOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  EPRConsumeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("dqc.epr_consume", odsAttrs.getContext());
  }

  EPRConsumeOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};
} // namespace detail
template <typename RangeT>
class EPRConsumeOpGenericAdaptor : public detail::EPRConsumeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::EPRConsumeOpGenericAdaptorBase;
  static constexpr int odsIndex_epr_handle = 0;
public:
  EPRConsumeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  EPRConsumeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : EPRConsumeOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  EPRConsumeOpGenericAdaptor(RangeT values, const EPRConsumeOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = EPRConsumeOp, typename = std::enable_if_t<std::is_same_v<LateInst, EPRConsumeOp>>>
  EPRConsumeOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getEprHandle() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class EPRConsumeOpAdaptor : public EPRConsumeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using EPRConsumeOpGenericAdaptor::EPRConsumeOpGenericAdaptor;
  EPRConsumeOpAdaptor(EPRConsumeOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class EPRConsumeOp : public ::mlir::Op<EPRConsumeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = EPRConsumeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = EPRConsumeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_epr_handle = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("dqc.epr_consume");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getEprHandle() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getEprHandleMutable() {
    auto range = getODSOperandIndexAndLength(0);
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

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value epr_handle);
  static EPRConsumeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value epr_handle);
  static EPRConsumeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value epr_handle);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value epr_handle);
  static EPRConsumeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value epr_handle);
  static EPRConsumeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value epr_handle);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static EPRConsumeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static EPRConsumeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace dqc
MLIR_DECLARE_EXPLICIT_TYPE_ID(::dqc::EPRConsumeOp)

namespace dqc {

//===----------------------------------------------------------------------===//
// ::dqc::PartitionInfoOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class PartitionInfoOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  PartitionInfoOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("dqc.partition_info", odsAttrs.getContext());
  }

  PartitionInfoOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::DictionaryAttr getPartitionMapAttr();
  ::mlir::DictionaryAttr getPartitionMap();
};
} // namespace detail
template <typename RangeT>
class PartitionInfoOpGenericAdaptor : public detail::PartitionInfoOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::PartitionInfoOpGenericAdaptorBase;
public:
  PartitionInfoOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  PartitionInfoOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : PartitionInfoOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  PartitionInfoOpGenericAdaptor(RangeT values, const PartitionInfoOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = PartitionInfoOp, typename = std::enable_if_t<std::is_same_v<LateInst, PartitionInfoOp>>>
  PartitionInfoOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

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
class PartitionInfoOpAdaptor : public PartitionInfoOpGenericAdaptor<::mlir::ValueRange> {
public:
  using PartitionInfoOpGenericAdaptor::PartitionInfoOpGenericAdaptor;
  PartitionInfoOpAdaptor(PartitionInfoOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class PartitionInfoOp : public ::mlir::Op<PartitionInfoOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PartitionInfoOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = PartitionInfoOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("partition_map")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getPartitionMapAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getPartitionMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("dqc.partition_info");
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

  ::mlir::DictionaryAttr getPartitionMapAttr() {
    return ::llvm::cast<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPartitionMapAttrName()));
  }

  ::mlir::DictionaryAttr getPartitionMap();
  void setPartitionMapAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getPartitionMapAttrName(), attr);
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::DictionaryAttr partition_map);
  static PartitionInfoOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::DictionaryAttr partition_map);
  static PartitionInfoOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::DictionaryAttr partition_map);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DictionaryAttr partition_map);
  static PartitionInfoOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::DictionaryAttr partition_map);
  static PartitionInfoOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::DictionaryAttr partition_map);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static PartitionInfoOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static PartitionInfoOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace dqc
MLIR_DECLARE_EXPLICIT_TYPE_ID(::dqc::PartitionInfoOp)

namespace dqc {

//===----------------------------------------------------------------------===//
// ::dqc::TeleGateMultiOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TeleGateMultiOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  TeleGateMultiOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("dqc.telegate_multi", odsAttrs.getContext());
  }

  TeleGateMultiOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getControlQpuAttr();
  uint32_t getControlQpu();
  ::mlir::ArrayAttr getTargetQpusAttr();
  ::mlir::ArrayAttr getTargetQpus();
};
} // namespace detail
template <typename RangeT>
class TeleGateMultiOpGenericAdaptor : public detail::TeleGateMultiOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TeleGateMultiOpGenericAdaptorBase;
  static constexpr int odsIndex_control_qubit = 0;
  static constexpr int odsIndex_target_qubits = 1;
  static constexpr int odsIndex_epr_handle = 2;
public:
  TeleGateMultiOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TeleGateMultiOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TeleGateMultiOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  TeleGateMultiOpGenericAdaptor(RangeT values, const TeleGateMultiOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = TeleGateMultiOp, typename = std::enable_if_t<std::is_same_v<LateInst, TeleGateMultiOp>>>
  TeleGateMultiOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getControlQubit() {
    return (*getODSOperands(0).begin());
  }

  RangeT getTargetQubits() {
    return getODSOperands(1);
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
class TeleGateMultiOpAdaptor : public TeleGateMultiOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TeleGateMultiOpGenericAdaptor::TeleGateMultiOpGenericAdaptor;
  TeleGateMultiOpAdaptor(TeleGateMultiOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TeleGateMultiOp : public ::mlir::Op<TeleGateMultiOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TeleGateMultiOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TeleGateMultiOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_control_qubit = 0;
  static constexpr int odsIndex_target_qubits = 1;
  static constexpr int odsIndex_epr_handle = 2;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("control_qpu"), ::llvm::StringRef("target_qpus")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getControlQpuAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getControlQpuAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getTargetQpusAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getTargetQpusAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("dqc.telegate_multi");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getControlQubit() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::Operation::operand_range getTargetQubits() {
    return getODSOperands(1);
  }

  ::mlir::TypedValue<::mlir::Type> getEprHandle() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getControlQubitMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getTargetQubitsMutable();
  ::mlir::OpOperand &getEprHandleMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::result_range getResults() {
    return getODSResults(0);
  }

  ::mlir::IntegerAttr getControlQpuAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getControlQpuAttrName()));
  }

  uint32_t getControlQpu();
  ::mlir::ArrayAttr getTargetQpusAttr() {
    return ::llvm::cast<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTargetQpusAttrName()));
  }

  ::mlir::ArrayAttr getTargetQpus();
  void setControlQpuAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getControlQpuAttrName(), attr);
  }

  void setControlQpu(uint32_t attrValue);
  void setTargetQpusAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTargetQpusAttrName(), attr);
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::Value control_qubit, ::mlir::ValueRange target_qubits, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::ArrayAttr target_qpus);
  static TeleGateMultiOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange results, ::mlir::Value control_qubit, ::mlir::ValueRange target_qubits, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::ArrayAttr target_qpus);
  static TeleGateMultiOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange results, ::mlir::Value control_qubit, ::mlir::ValueRange target_qubits, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::ArrayAttr target_qpus);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::Value control_qubit, ::mlir::ValueRange target_qubits, ::mlir::Value epr_handle, uint32_t control_qpu, ::mlir::ArrayAttr target_qpus);
  static TeleGateMultiOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange results, ::mlir::Value control_qubit, ::mlir::ValueRange target_qubits, ::mlir::Value epr_handle, uint32_t control_qpu, ::mlir::ArrayAttr target_qpus);
  static TeleGateMultiOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange results, ::mlir::Value control_qubit, ::mlir::ValueRange target_qubits, ::mlir::Value epr_handle, uint32_t control_qpu, ::mlir::ArrayAttr target_qpus);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TeleGateMultiOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TeleGateMultiOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
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
} // namespace dqc
MLIR_DECLARE_EXPLICIT_TYPE_ID(::dqc::TeleGateMultiOp)

namespace dqc {

//===----------------------------------------------------------------------===//
// ::dqc::TeleGateOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TeleGateOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  TeleGateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("dqc.telegate", odsAttrs.getContext());
  }

  TeleGateOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getControlQpuAttr();
  uint32_t getControlQpu();
  ::mlir::IntegerAttr getTargetQpuAttr();
  uint32_t getTargetQpu();
};
} // namespace detail
template <typename RangeT>
class TeleGateOpGenericAdaptor : public detail::TeleGateOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TeleGateOpGenericAdaptorBase;
  static constexpr int odsIndex_control_qubit = 0;
  static constexpr int odsIndex_target_qubit = 1;
  static constexpr int odsIndex_epr_handle = 2;
public:
  TeleGateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TeleGateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TeleGateOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  TeleGateOpGenericAdaptor(RangeT values, const TeleGateOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = TeleGateOp, typename = std::enable_if_t<std::is_same_v<LateInst, TeleGateOp>>>
  TeleGateOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getControlQubit() {
    return (*getODSOperands(0).begin());
  }

  ValueT getTargetQubit() {
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
class TeleGateOpAdaptor : public TeleGateOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TeleGateOpGenericAdaptor::TeleGateOpGenericAdaptor;
  TeleGateOpAdaptor(TeleGateOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TeleGateOp : public ::mlir::Op<TeleGateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TeleGateOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TeleGateOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_control_qubit = 0;
  static constexpr int odsIndex_target_qubit = 1;
  static constexpr int odsIndex_epr_handle = 2;
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
    return ::llvm::StringLiteral("dqc.telegate");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getControlQubit() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getTargetQubit() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getEprHandle() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getControlQubitMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getTargetQubitMutable() {
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

  ::mlir::TypedValue<::mlir::Type> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSResults(0).begin());
  }

  ::mlir::IntegerAttr getControlQpuAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getControlQpuAttrName()));
  }

  uint32_t getControlQpu();
  ::mlir::IntegerAttr getTargetQpuAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTargetQpuAttrName()));
  }

  uint32_t getTargetQpu();
  void setControlQpuAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getControlQpuAttrName(), attr);
  }

  void setControlQpu(uint32_t attrValue);
  void setTargetQpuAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getTargetQpuAttrName(), attr);
  }

  void setTargetQpu(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu);
  static TeleGateOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu);
  static TeleGateOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu);
  static TeleGateOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu);
  static TeleGateOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu);
  static TeleGateOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu);
  static TeleGateOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu);
  static TeleGateOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu);
  static TeleGateOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TeleGateOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TeleGateOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
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
} // namespace dqc
MLIR_DECLARE_EXPLICIT_TYPE_ID(::dqc::TeleGateOp)


#endif  // GET_OP_CLASSES


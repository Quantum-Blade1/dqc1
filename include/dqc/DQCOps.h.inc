/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)
#undef GET_OP_FWD_DEFINES
namespace dqc {
class EPRAllocOp;
} // namespace dqc
namespace dqc {
class EPRConsumeOp;
} // namespace dqc
namespace dqc {
class PartitionInfoOp;
} // namespace dqc
namespace dqc {
class TeleGateMultiOp;
} // namespace dqc
namespace dqc {
class TeleGateOp;
} // namespace dqc
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace dqc {

//===----------------------------------------------------------------------===//
// ::dqc::EPRAllocOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class EPRAllocOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  EPRAllocOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
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
  EPRAllocOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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

  ::mlir::LogicalResult verify(::mlir::Location loc);
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

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getEprHandle();
  ::mlir::IntegerAttr getSourceQpuAttr();
  uint32_t getSourceQpu();
  ::mlir::IntegerAttr getTargetQpuAttr();
  uint32_t getTargetQpu();
  ::mlir::FloatAttr getFidelityAttr();
  ::std::optional< ::llvm::APFloat > getFidelity();
  void setSourceQpuAttr(::mlir::IntegerAttr attr);
  void setSourceQpu(uint32_t attrValue);
  void setTargetQpuAttr(::mlir::IntegerAttr attr);
  void setTargetQpu(uint32_t attrValue);
  void setFidelityAttr(::mlir::FloatAttr attr);
  void setFidelity(::std::optional<::llvm::APFloat> attrValue);
  ::mlir::Attribute removeFidelityAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type epr_handle, ::mlir::IntegerAttr source_qpu, ::mlir::IntegerAttr target_qpu, /*optional*/::mlir::FloatAttr fidelity);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr source_qpu, ::mlir::IntegerAttr target_qpu, /*optional*/::mlir::FloatAttr fidelity);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type epr_handle, uint32_t source_qpu, uint32_t target_qpu, /*optional*/::mlir::FloatAttr fidelity);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t source_qpu, uint32_t target_qpu, /*optional*/::mlir::FloatAttr fidelity);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
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
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  EPRConsumeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class EPRConsumeOpGenericAdaptor : public detail::EPRConsumeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::EPRConsumeOpGenericAdaptorBase;
public:
  EPRConsumeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getEprHandle() {
    return *getODSOperands(0).begin();
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

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class EPRConsumeOp : public ::mlir::Op<EPRConsumeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = EPRConsumeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = EPRConsumeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("dqc.epr_consume");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getEprHandle();
  ::mlir::MutableOperandRange getEprHandleMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value epr_handle);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value epr_handle);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
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
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  PartitionInfoOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DictionaryAttr getPartitionMapAttr();
  ::mlir::DictionaryAttr getPartitionMap();
};
} // namespace detail
template <typename RangeT>
class PartitionInfoOpGenericAdaptor : public detail::PartitionInfoOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::PartitionInfoOpGenericAdaptorBase;
public:
  PartitionInfoOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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

  ::mlir::LogicalResult verify(::mlir::Location loc);
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

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DictionaryAttr getPartitionMapAttr();
  ::mlir::DictionaryAttr getPartitionMap();
  void setPartitionMapAttr(::mlir::DictionaryAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::DictionaryAttr partition_map);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DictionaryAttr partition_map);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
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
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  TeleGateMultiOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
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
public:
  TeleGateMultiOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getControlQubit() {
    return *getODSOperands(0).begin();
  }

  RangeT getTargetQubits() {
    return getODSOperands(1);
  }

  ValueT getEprHandle() {
    return *getODSOperands(2).begin();
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

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class TeleGateMultiOp : public ::mlir::Op<TeleGateMultiOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TeleGateMultiOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TeleGateMultiOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
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
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getControlQubit();
  ::mlir::Operation::operand_range getTargetQubits();
  ::mlir::Value getEprHandle();
  ::mlir::MutableOperandRange getControlQubitMutable();
  ::mlir::MutableOperandRange getTargetQubitsMutable();
  ::mlir::MutableOperandRange getEprHandleMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::IntegerAttr getControlQpuAttr();
  uint32_t getControlQpu();
  ::mlir::ArrayAttr getTargetQpusAttr();
  ::mlir::ArrayAttr getTargetQpus();
  void setControlQpuAttr(::mlir::IntegerAttr attr);
  void setControlQpu(uint32_t attrValue);
  void setTargetQpusAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::Value control_qubit, ::mlir::ValueRange target_qubits, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::ArrayAttr target_qpus);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::Value control_qubit, ::mlir::ValueRange target_qubits, ::mlir::Value epr_handle, uint32_t control_qpu, ::mlir::ArrayAttr target_qpus);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
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
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  TeleGateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
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
public:
  TeleGateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getControlQubit() {
    return *getODSOperands(0).begin();
  }

  ValueT getTargetQubit() {
    return *getODSOperands(1).begin();
  }

  ValueT getEprHandle() {
    return *getODSOperands(2).begin();
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

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class TeleGateOp : public ::mlir::Op<TeleGateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TeleGateOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TeleGateOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
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

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getControlQubit();
  ::mlir::Value getTargetQubit();
  ::mlir::Value getEprHandle();
  ::mlir::MutableOperandRange getControlQubitMutable();
  ::mlir::MutableOperandRange getTargetQubitMutable();
  ::mlir::MutableOperandRange getEprHandleMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getResult();
  ::mlir::IntegerAttr getControlQpuAttr();
  uint32_t getControlQpu();
  ::mlir::IntegerAttr getTargetQpuAttr();
  uint32_t getTargetQpu();
  void setControlQpuAttr(::mlir::IntegerAttr attr);
  void setControlQpu(uint32_t attrValue);
  void setTargetQpuAttr(::mlir::IntegerAttr attr);
  void setTargetQpu(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, ::mlir::IntegerAttr control_qpu, ::mlir::IntegerAttr target_qpu);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value control_qubit, ::mlir::Value target_qubit, ::mlir::Value epr_handle, uint32_t control_qpu, uint32_t target_qpu);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace dqc
MLIR_DECLARE_EXPLICIT_TYPE_ID(::dqc::TeleGateOp)


#endif  // GET_OP_CLASSES


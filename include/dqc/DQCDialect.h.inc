/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: DQCDialect.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace dqc {

/// The DQC (Distributed Quantum Computing) dialect provides operations for
///     managing distributed quantum computations across multiple QPUs, including
///     entanglement allocation and teleported gate sequences.
class DQCDialect : public ::mlir::Dialect {
  explicit DQCDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~DQCDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("dqc");
  }

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;

  /// Materialize a single constant operation from a given attribute value with
  /// the desired resultant type.
  ::mlir::Operation *materializeConstant(::mlir::OpBuilder &builder,
                                         ::mlir::Attribute value,
                                         ::mlir::Type type,
                                         ::mlir::Location loc) override;
};
} // namespace dqc
MLIR_DECLARE_EXPLICIT_TYPE_ID(::dqc::DQCDialect)

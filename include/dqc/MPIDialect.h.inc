/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: MPIDialect.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace dqc::mpi {

/// Dialect for distributed quantum communication primitives
class MPIDialect : public ::mlir::Dialect {
  explicit MPIDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~MPIDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("mpi");
  }
};

} // namespace dqc::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::dqc::mpi::MPIDialect)

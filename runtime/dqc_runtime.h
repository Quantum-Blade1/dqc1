//===- dqc_runtime.h - DQC Quantum Simulator Runtime ----*- C -*-===//
//
// Public API for the state-vector quantum simulator. Generated LLVM IR
// from the DQC compiler calls into these functions at runtime.
//
//===-----------------------------------------------------------------===//

#ifndef DQC_RUNTIME_H
#define DQC_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

// -- State management -----------------------------------------------

void dqc_init(int num_qubits);
void dqc_finalize(void);
int  dqc_alloc_qubit(void);

// -- Single-qubit gates ---------------------------------------------

void dqc_h(int q);
void dqc_x(int q);
void dqc_y(int q);
void dqc_z(int q);
void dqc_s(int q);
void dqc_t(int q);
void dqc_rx(int q, double angle);
void dqc_ry(int q, double angle);
void dqc_rz(int q, double angle);

// -- Multi-qubit gates ----------------------------------------------

void dqc_cnot(int ctrl, int tgt);
void dqc_cz(int ctrl, int tgt);
void dqc_swap(int q0, int q1);
void dqc_ccx(int c0, int c1, int tgt);

// -- Measurement ----------------------------------------------------

int dqc_measure(int q);

// -- Reset ----------------------------------------------------------

void dqc_reset(int q);

// -- Distributed / MPI stubs ----------------------------------------

void dqc_distribute_epr(int src_qpu, int tgt_qpu, int *epr_id);
void dqc_telegate_sequence(int ctrl, int tgt, int epr_id,
                           int ctrl_qpu, int tgt_qpu);

// -- Debug / verbosity ----------------------------------------------

void dqc_dump_state(void);
void dqc_set_verbose(int level);

#ifdef __cplusplus
}
#endif

#endif // DQC_RUNTIME_H

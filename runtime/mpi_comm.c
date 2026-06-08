//===- mpi_comm.c - Simulated distributed quantum ops -----*- C -*-===//
//
// Single-node stubs for EPR distribution and telegate sequencing.
// On a real multi-node deployment these would issue MPI_Send/Recv;
// here they delegate to local state-vector operations.
//
//===--------------------------------------------------------------===//

#include "dqc_runtime.h"
#include <stdio.h>

extern int *dqc_get_next_epr_id(void);
extern int  dqc_is_verbose(void);

void dqc_distribute_epr(int src_qpu, int tgt_qpu, int *epr_id) {
    int *next = dqc_get_next_epr_id();
    *epr_id = (*next)++;
    if (dqc_is_verbose())
        printf("[dqc] EPR pair #%d: QPU %d <-> QPU %d\n",
               *epr_id, src_qpu, tgt_qpu);
}

void dqc_telegate_sequence(int ctrl, int tgt, int epr_id,
                           int ctrl_qpu, int tgt_qpu) {
    if (dqc_is_verbose())
        printf("[dqc] telegate q%d -> q%d via EPR #%d (QPU %d -> QPU %d)\n",
               ctrl, tgt, epr_id, ctrl_qpu, tgt_qpu);
    dqc_cnot(ctrl, tgt);
}

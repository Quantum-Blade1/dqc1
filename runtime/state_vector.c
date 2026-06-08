//===- state_vector.c - Dense state-vector management -----*- C -*-===//
//
// Manages the 2^n complex amplitude array that represents the quantum
// state. Provides allocation, initialisation, teardown, and a visual
// state dump with probability bars.
//
//===-----------------------------------------------------------------===//

#include "dqc_runtime.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_num_qubits = 0;
static int g_next_qubit = 0;
static long g_dim       = 0;
static double complex *g_state = NULL;

static int g_next_epr_id = 0;
static int g_verbose     = 0;

/* Internal accessors used by gates.c, measurement.c, mpi_comm.c */
double complex *dqc_get_state(void)    { return g_state; }
long            dqc_get_dim(void)      { return g_dim; }
int             dqc_get_num_qubits(void) { return g_num_qubits; }
int            *dqc_get_next_epr_id(void) { return &g_next_epr_id; }
int             dqc_is_verbose(void)   { return g_verbose; }

void dqc_set_verbose(int level) { g_verbose = level; }

/* ------------------------------------------------------------------ */

void dqc_init(int num_qubits) {
    g_num_qubits  = num_qubits;
    g_next_qubit  = 0;
    g_dim         = 1L << num_qubits;
    g_next_epr_id = 0;

    const char *v = getenv("DQC_VERBOSE");
    if (v && v[0] == '1')
        g_verbose = 1;

    g_state = (double complex *)calloc(g_dim, sizeof(double complex));
    if (!g_state) {
        fprintf(stderr,
                "dqc: failed to allocate state vector for %d qubits\n",
                num_qubits);
        exit(1);
    }
    g_state[0] = 1.0 + 0.0 * I;

    if (g_verbose)
        printf("[dqc] initialized %d-qubit simulator (%ld amplitudes)\n",
               num_qubits, g_dim);
}

void dqc_finalize(void) {
    free(g_state);
    g_state      = NULL;
    g_dim        = 0;
    g_num_qubits = 0;
    g_next_qubit = 0;
}

int dqc_alloc_qubit(void) {
    if (g_next_qubit >= g_num_qubits) {
        fprintf(stderr, "dqc: qubit allocation overflow (max %d)\n",
                g_num_qubits);
        exit(1);
    }
    return g_next_qubit++;
}

/* ------------------------------------------------------------------ */

void dqc_dump_state(void) {
    printf("\n");
    printf("  Quantum State  (%d qubits)\n", g_num_qubits);
    printf("  ─────────────────────────────────────\n");

    int printed = 0;
    for (long i = 0; i < g_dim; i++) {
        double re   = creal(g_state[i]);
        double im   = cimag(g_state[i]);
        double prob = re * re + im * im;

        if (prob < 1e-10)
            continue;

        printf("  |");
        for (int b = g_num_qubits - 1; b >= 0; b--)
            printf("%d", (int)((i >> b) & 1));
        printf(">  ");

        int bar = (int)(prob * 40 + 0.5);
        if (bar < 1) bar = 1;
        for (int b = 0; b < bar; b++)
            printf("█");
        printf("  %.1f%%", prob * 100.0);

        if (g_verbose)
            printf("   (%+.4f %+.4fi)", re, im);

        printf("\n");
        printed++;
    }

    if (!printed)
        printf("  (zero state)\n");
    printf("\n");
}

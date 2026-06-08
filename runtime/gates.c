//===- gates.c - Quantum gate implementations -----------*- C -*-===//
//
// Unitary transformations on a dense state vector.
// Qubit 0 is the least-significant bit of the basis-state index.
//
//===-------------------------------------------------------------===//

#include "dqc_runtime.h"
#include <complex.h>
#include <math.h>

extern double complex *dqc_get_state(void);
extern long            dqc_get_dim(void);

/* ------------------------------------------------------------------ */
/* Single-qubit gate helper                                           */
/* ------------------------------------------------------------------ */

static void apply_gate(int q,
                       double complex u00, double complex u01,
                       double complex u10, double complex u11) {
    double complex *sv = dqc_get_state();
    long dim  = dqc_get_dim();
    long mask = 1L << q;

    for (long i = 0; i < dim; i++) {
        if (i & mask) continue;          /* process pairs with bit q = 0 */
        long j = i | mask;

        double complex a = sv[i];
        double complex b = sv[j];
        sv[i] = u00 * a + u01 * b;
        sv[j] = u10 * a + u11 * b;
    }
}

/* ------------------------------------------------------------------ */
/* Single-qubit gates                                                 */
/* ------------------------------------------------------------------ */

void dqc_h(int q) {
    double s = 1.0 / sqrt(2.0);
    apply_gate(q, s, s, s, -s);
}

void dqc_x(int q) { apply_gate(q, 0, 1, 1, 0); }
void dqc_y(int q) { apply_gate(q, 0, -I, I, 0); }
void dqc_z(int q) { apply_gate(q, 1, 0, 0, -1); }
void dqc_s(int q) { apply_gate(q, 1, 0, 0, I); }

void dqc_t(int q) {
    apply_gate(q, 1, 0, 0, cexp(I * M_PI / 4.0));
}

void dqc_rx(int q, double angle) {
    double c = cos(angle / 2.0), s = sin(angle / 2.0);
    apply_gate(q, c, -I * s, -I * s, c);
}

void dqc_ry(int q, double angle) {
    double c = cos(angle / 2.0), s = sin(angle / 2.0);
    apply_gate(q, c, -s, s, c);
}

void dqc_rz(int q, double angle) {
    apply_gate(q, cexp(-I * angle / 2.0), 0,
                  0, cexp(I * angle / 2.0));
}

/* ------------------------------------------------------------------ */
/* Two-qubit gates                                                    */
/* ------------------------------------------------------------------ */

void dqc_cnot(int ctrl, int tgt) {
    double complex *sv = dqc_get_state();
    long dim   = dqc_get_dim();
    long cmask = 1L << ctrl;
    long tmask = 1L << tgt;

    for (long i = 0; i < dim; i++) {
        if ((i & cmask) && !(i & tmask)) {
            long j = i | tmask;
            double complex tmp = sv[i];
            sv[i] = sv[j];
            sv[j] = tmp;
        }
    }
}

void dqc_cz(int ctrl, int tgt) {
    double complex *sv = dqc_get_state();
    long dim   = dqc_get_dim();
    long cmask = 1L << ctrl;
    long tmask = 1L << tgt;

    for (long i = 0; i < dim; i++)
        if ((i & cmask) && (i & tmask))
            sv[i] = -sv[i];
}

void dqc_swap(int q0, int q1) {
    double complex *sv = dqc_get_state();
    long dim = dqc_get_dim();
    long m0  = 1L << q0;
    long m1  = 1L << q1;

    for (long i = 0; i < dim; i++) {
        int b0 = (i & m0) ? 1 : 0;
        int b1 = (i & m1) ? 1 : 0;
        if (b0 != b1 && b0 == 0) {
            long j = i ^ m0 ^ m1;
            double complex tmp = sv[i];
            sv[i] = sv[j];
            sv[j] = tmp;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Three-qubit gates                                                  */
/* ------------------------------------------------------------------ */

void dqc_ccx(int c0, int c1, int tgt) {
    double complex *sv = dqc_get_state();
    long dim = dqc_get_dim();
    long m0  = 1L << c0;
    long m1  = 1L << c1;
    long mt  = 1L << tgt;

    for (long i = 0; i < dim; i++) {
        if ((i & m0) && (i & m1) && !(i & mt)) {
            long j = i | mt;
            double complex tmp = sv[i];
            sv[i] = sv[j];
            sv[j] = tmp;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Reset                                                              */
/* ------------------------------------------------------------------ */

extern int dqc_measure(int q);

void dqc_reset(int q) {
    int outcome = dqc_measure(q);
    if (outcome == 1)
        dqc_x(q);
}

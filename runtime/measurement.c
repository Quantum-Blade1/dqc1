//===- measurement.c - Projective measurement + collapse --*- C -*-===//
//===--------------------------------------------------------------===//

#include "dqc_runtime.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

extern double complex *dqc_get_state(void);
extern long            dqc_get_dim(void);
extern int             dqc_is_verbose(void);

static int g_rng_seeded = 0;

int dqc_measure(int q) {
    /* Seed the PRNG once with sub-second entropy. */
    if (!g_rng_seeded) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        srand((unsigned)(tv.tv_sec ^ tv.tv_usec ^ (long)&tv));
        g_rng_seeded = 1;
    }

    double complex *sv = dqc_get_state();
    long dim  = dqc_get_dim();
    long mask = 1L << q;

    /* Probability of outcome |1>. */
    double prob1 = 0.0;
    for (long i = 0; i < dim; i++) {
        if (i & mask) {
            double re = creal(sv[i]), im = cimag(sv[i]);
            prob1 += re * re + im * im;
        }
    }

    double r = (double)rand() / (double)RAND_MAX;
    int outcome = (r < prob1) ? 1 : 0;

    /* Collapse: zero out incompatible amplitudes, then renormalise. */
    double norm = 0.0;
    for (long i = 0; i < dim; i++) {
        int bit = (i & mask) ? 1 : 0;
        if (bit != outcome) {
            sv[i] = 0.0;
        } else {
            double re = creal(sv[i]), im = cimag(sv[i]);
            norm += re * re + im * im;
        }
    }

    if (norm > 1e-15) {
        double scale = 1.0 / sqrt(norm);
        for (long i = 0; i < dim; i++)
            sv[i] *= scale;
    }

    if (dqc_is_verbose())
        printf("[dqc] measured qubit %d = %d (prob was %.4f)\n",
               q, outcome, outcome ? prob1 : 1.0 - prob1);
    return outcome;
}

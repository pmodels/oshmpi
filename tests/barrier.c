/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>

#define ITER 3

int main(int argc, char *argv[])
{
    int mype, npes;
    static long pSync[SHMEM_BARRIER_SYNC_SIZE];
    int i, x;

    for (i = 0; i < SHMEM_BARRIER_SYNC_SIZE; i++)
        pSync[i] = SHMEM_SYNC_VALUE;

    shmem_init();

    mype = shmem_my_pe();
    npes = shmem_n_pes();

    for (x = 0; x < ITER; x++) {
        if (mype % 2 == 0) {
            /* synchronize all even pes */
            shmem_barrier(0, 1, (npes / 2 + npes % 2), pSync);
            fprintf(stdout, "I am pe %d in %d, barrier on even pes\n", mype, npes);
            fflush(stdout);
        }

        if (mype % 2 == 1) {
            /* synchronize all odd pes */
            shmem_barrier(1, 1, (npes / 2), pSync);
            fprintf(stdout, "I am pe %d in %d, barrier on odd pes\n", mype, npes);
            fflush(stdout);
        }

        if (mype < npes - 1) {
            /* synchronize all (npes-1) pes */
            shmem_barrier(0, 0, npes - 1, pSync);
            fprintf(stdout, "I am pe %d in %d, barrier on (npes-1) pes\n", mype, npes);
            fflush(stdout);
        }

        if (mype < npes - 2) {
            /* synchronize all (npes-2) pes */
            shmem_barrier(0, 0, npes - 2, pSync);
            fprintf(stdout, "I am pe %d in %d, barrier on (npes-2) pes\n", mype, npes);
            fflush(stdout);
        }

        if (mype < 2) {
            /* synchronize the first two pes */
            shmem_barrier(0, 0, 2, pSync);
            fprintf(stdout, "I am pe %d in %d, barrier on 0-1 pes\n", mype, npes);
            fflush(stdout);
        }

        /* synchronize the first two pes */
        shmem_barrier(mype, 0, 1, pSync);
        fprintf(stdout, "I am pe %d in %d, barrier on me\n", mype, npes);
        fflush(stdout);

        /* synchronize all pes */
        shmem_barrier(0, 0, npes, pSync);
        fprintf(stdout, "I am pe %d in %d, barrier on all pes\n", mype, npes);
        fflush(stdout);
    }

    shmem_finalize();
    return 0;
}

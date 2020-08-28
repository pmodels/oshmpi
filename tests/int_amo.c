/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>

#define ITER 10

int main(int argc, char *argv[])
{
    int mype;
#ifdef USE_SYMM_HEAP
    int *dst;
#else
    static int dst[1];
#endif
    int x, errs = 0;

    shmem_init();
    mype = shmem_my_pe();

#ifdef USE_SYMM_HEAP
    dst = (int *) shmem_malloc(sizeof(int));
#endif

    dst[0] = 0;
    shmem_barrier_all();

    for (x = 0; x < ITER; x++) {
        if (mype == 0) {
            int oldval = shmem_int_atomic_fetch_inc(dst, 1);
            if (oldval != x) {
                fprintf(stderr, "Excepted oldval %d at iter %d, but %d\n", x, x, oldval);
                fflush(stderr);
                errs++;
            }
        }
    }

    shmem_barrier_all();

    if (mype == 1) {
        if (dst[0] != ITER) {
            fprintf(stderr, "Excepted dst %d, but %d\n", ITER, dst[0]);
            fflush(stderr);
            errs++;
        }
    }
#ifdef USE_SYMM_HEAP
    shmem_free(dst);
#endif

    shmem_finalize();

    if (mype == 1 && errs == 0) {
        fprintf(stdout, "Passed\n");
        fflush(stdout);
    }

    return 0;
}

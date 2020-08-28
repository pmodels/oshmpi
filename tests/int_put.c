/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>

#define SIZE 10
#define ITER 10

int main(int argc, char *argv[])
{
    int mype;
#ifdef USE_SYMM_HEAP
    int *dst, *src;
#else
    static int dst[SIZE * ITER], src[SIZE * ITER];
#endif
    int i, x, errs = 0;

    shmem_init();

    mype = shmem_my_pe();

#ifdef USE_SYMM_HEAP
    dst = (int *) shmem_malloc(SIZE * ITER * sizeof(int));
    src = (int *) shmem_malloc(SIZE * ITER * sizeof(int));
#endif

    for (i = 0; i < SIZE * ITER; i++) {
        src[i] = mype + i;
        dst[i] = 0;
    }
    shmem_barrier_all();

    for (x = 0; x < ITER; x++) {
        int off = x * SIZE;
        if (mype == 0) {
            shmem_int_put(&dst[off], &src[off], SIZE, 1);
            shmem_quiet();
        }
    }

    shmem_barrier_all();

    if (mype == 1) {
        for (i = 0; i < SIZE * ITER; i++) {
            if (dst[i] != i) {
                fprintf(stderr, "Excepted %d at dst[%d], but %d\n", i, i, dst[i]);
                fflush(stderr);
                errs++;
            }
        }
    }
#ifdef USE_SYMM_HEAP
    shmem_free(dst);
    shmem_free(src);
#endif

    shmem_finalize();

    if (mype == 1 && errs == 0) {
        fprintf(stdout, "Passed\n");
        fflush(stdout);
    }

    return 0;
}

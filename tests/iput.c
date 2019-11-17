/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>

#define SIZE 512        /* buffer size in number of elements
                         * stride * nelems should not exceed the limit*/
#define ITER 3  /* repeat the same strided pattern to test datatype cache */
static int mype, npes;

static void runtest(ptrdiff_t dst, ptrdiff_t sst, size_t nelems)
{
    int *loc_buf = NULL, *remote_buf = NULL;
    int i;

    loc_buf = malloc(SIZE * sizeof(int));
    remote_buf = shmem_malloc(SIZE * sizeof(int));

    for (i = 0; i < SIZE; i++) {
        loc_buf[i] = mype * npes + i;
        remote_buf[i] = 0;
    }

    shmem_sync_all();
    if (mype == 0) {
        int x;
        for (x = 0; x < ITER; x++)
            shmem_int_iput(remote_buf, loc_buf, dst, sst, nelems, 1);
        shmem_quiet();
    }
    shmem_sync_all();

    if (mype == 1) {
        int elem_idx = 0;
        for (i = 0; i < SIZE; i++) {
            int exp_val = 0;
            if (i % dst == 0 && elem_idx < nelems) {
                exp_val = elem_idx * sst;
                elem_idx++;
            }
            if (remote_buf[i] != exp_val) {
                fprintf(stderr, "[%d] Expected remote_buf[%d] %d but received %d"
                        "[dst=%ld, sst=%ld, nelems=%ld]\n",
                        mype, i, exp_val, remote_buf[i], dst, sst, nelems);
                fflush(stderr);

                /* do not continue if any error is found */
                int j;
                for (j = 0; j < SIZE; j++)
                    fprintf(stderr, "%d ", remote_buf[j]);
                fprintf(stderr, "\n");
                fflush(stderr);
                shmem_global_exit(-1);
            }
        }
    }

    shmem_free(remote_buf);
    free(loc_buf);

    if (mype == 0) {
        fprintf(stdout, "Test [dst=%ld, sst=%ld, nelems=%ld] done\n", dst, sst, nelems);
        fflush(stdout);
    }
}

int main(int argc, char *argv[])
{
    shmem_init();

    mype = shmem_my_pe();
    npes = shmem_n_pes();

    /* Create different strided datatypes to test datatype cache. */
    runtest(512, 4, 1);
    runtest(256, 4, 2);
    runtest(128, 4, 4);
    runtest(64, 4, 8);
    runtest(32, 4, 16);
    runtest(16, 4, 32);
    runtest(8, 4, 64);
    runtest(4, 1, 128);
    runtest(2, 1, 256);
    runtest(2, 2, 256);
    runtest(1, 2, 256);
    runtest(1, 1, 512);

    shmem_finalize();

    if (mype == 0) {
        fprintf(stdout, "Passed\n");
        fflush(stderr);
    }

    return 0;
}

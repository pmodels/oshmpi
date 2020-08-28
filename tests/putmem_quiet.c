/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>

#define SIZE 128
#define ITER 5

#ifdef ENABLE_TRACING_SSC
#define TRACING_SSC_MARK(MARK_ID)                       \
        __asm__ __volatile__ (\
                "\n\t  movl $"#MARK_ID", %%ebx"         \
                "\n\t  .byte 0x64, 0x67, 0x90"          \
                : : : "%ebx","memory");
#else
#define TRACING_SSC_MARK(MARK_ID)
#endif

int main(int argc, char *argv[])
{
    int mype, npes;
    int *loc_buf = NULL;
#ifdef TEST_SYMM_TEXT
    static int remote_buf[SIZE];
#else
    int *remote_buf = NULL;
#endif
    int i;

    shmem_init();

    mype = shmem_my_pe();
    npes = shmem_n_pes();

    loc_buf = malloc(SIZE * sizeof(int));
#ifndef TEST_SYMM_TEXT
    remote_buf = shmem_malloc(SIZE * sizeof(int));
#endif

    for (i = 0; i < SIZE; i++) {
        loc_buf[i] = mype * npes + i;
        remote_buf[i] = 0;
    }

    shmem_sync_all();

    if (mype == 0) {
        int x;
        for (x = 0; x < ITER; x++) {
            TRACING_SSC_MARK(0x4000);   /* starting_flag */
            shmem_putmem(remote_buf, loc_buf, SIZE * sizeof(int), 1);
            shmem_quiet();
            TRACING_SSC_MARK(0x4100);   /* ending_flag */
        }
    }

    shmem_sync_all();

    if (mype == 1) {
        for (i = 0; i < SIZE; i++) {
            if (remote_buf[i] != i) {
                fprintf(stderr, "[%d] Expected remote_buf[%d] %d but received %d\n",
                        mype, i, i, remote_buf[i]);
                fflush(stderr);
                shmem_global_exit(-1);
            }
        }
    }
#ifndef TEST_SYMM_TEXT
    shmem_free(remote_buf);
#endif
    free(loc_buf);

    shmem_finalize();

    if (mype == 0) {
        fprintf(stdout, "Passed\n");
        fflush(stderr);
    }

    return 0;
}

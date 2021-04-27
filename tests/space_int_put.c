/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>
#include <shmemx.h>
#include "gpu_common.h"

#define SIZE 10
#define ITER 10

int main(int argc, char *argv[])
{
    int mype, errs = 0;
    int x;

    shmem_init();
    mype = shmem_my_pe();

    void *device_handle;
    init_device(mype, &device_handle);

    shmemx_space_config_t space_config;
    shmemx_space_t space;

    space_config.sheap_size = 1 << 20;
    space_config.num_contexts = 0;
    space_config.hints = 0;
#ifdef USE_CUDA
    space_config.memkind = SHMEMX_MEM_CUDA;
#else
    space_config.memkind = SHMEMX_MEM_HOST;
#endif
    shmemx_space_create(space_config, &space);
    shmemx_space_attach(space);

    int *src = shmemx_space_malloc(space, SIZE * ITER * sizeof(int));
    int *dst = shmemx_space_malloc(space, SIZE * ITER * sizeof(int));

    reset_data(mype, SIZE, ITER, src, dst);
    shmem_barrier_all();

    for (x = 0; x < ITER; x++) {
        int off = x * SIZE;
        if (mype == 0) {
            shmem_int_put(&dst[off], &src[off], SIZE, 1);
            shmem_quiet();
        }
    }

    shmem_barrier_all();

    if (mype == 1)
        errs += check_data(SIZE, ITER, dst);

    shmem_free(dst);
    shmem_free(src);

    shmemx_space_detach(space);
    shmemx_space_destroy(space);
    shmem_finalize();

    if (mype == 1 && errs == 0) {
        fprintf(stdout, "Passed\n");
        fflush(stderr);
    }
    return 0;
}

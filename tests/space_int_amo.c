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

#define ITER 10

int mype, errs = 0;

#ifdef USE_CUDA
#include <cuda_runtime_api.h>

static void reset_amo_data(int *dst)
{
    cudaMemset(dst, 0, sizeof(int));
}

static void check_amo_data(int *dst)
{
    int tmpbuf;
    cudaMemcpy(&tmpbuf, dst, sizeof(int), cudaMemcpyDeviceToHost);
    if (tmpbuf != ITER) {
        fprintf(stderr, "Excepted dst %d, but %d\n", ITER, tmpbuf);
        fflush(stderr);
        errs++;
    }
}
#else
static void reset_amo_data(int *dst)
{
    dst[0] = 0;
}

static void check_amo_data(int *dst)
{
    if (dst[0] != ITER) {
        fprintf(stderr, "Excepted dst %d, but %d\n", ITER, dst[0]);
        fflush(stderr);
        errs++;
    }
}
#endif

int main(int argc, char *argv[])
{
    int x;

    shmem_init();
    mype = shmem_my_pe();

    void *device_handle;
    init_device(mype, &device_handle);

    shmemx_space_config_t space_config;
    shmemx_space_t space;

    space_config.sheap_size = 1 << 20;
    space_config.num_contexts = 0;
#ifdef USE_CUDA
    space_config.memkind = SHMEMX_MEM_CUDA;
#else
    space_config.memkind = SHMEMX_MEM_HOST;
#endif
    shmemx_space_create(space_config, &space);
    shmemx_space_attach(space);

    int *dst = shmemx_space_malloc(space, sizeof(int));

    reset_amo_data(dst);
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

    if (mype == 1)
        check_amo_data(dst);

    shmem_free(dst);
    shmemx_space_detach(space);
    shmem_finalize();

    if (mype == 1 && errs == 0) {
        fprintf(stdout, "Passed\n");
        fflush(stderr);
    }
    return 0;
}

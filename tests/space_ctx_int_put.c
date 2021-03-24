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
int mype, errs = 0;

#ifdef USE_CUDA
#include <cuda_runtime_api.h>

static void reset_data(int *src, int *dst)
{
    int tmpbuf[SIZE * ITER];
    int i;

    for (i = 0; i < SIZE * ITER; i++)
        tmpbuf[i] = mype + i;
    cudaMemcpy(src, tmpbuf, SIZE * ITER * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dst, 0, SIZE * ITER * sizeof(int));
}

static void check_data(int *dst)
{
    int tmpbuf[SIZE * ITER], i;
    cudaMemcpy(tmpbuf, dst, SIZE * ITER * sizeof(int), cudaMemcpyDeviceToHost);
    for (i = 0; i < SIZE * ITER; i++) {
        if (tmpbuf[i] != i) {
            fprintf(stderr, "Excepted %d at dst[%d], but %d\n", i, i, tmpbuf[i]);
            fflush(stderr);
            errs++;
        }
    }
}
#else
static void reset_data(int *src, int *dst)
{
    int i;
    for (i = 0; i < SIZE * ITER; i++) {
        src[i] = mype + i;
        dst[i] = 0;
    }
}

static void check_data(int *dst)
{
    int i;
    for (i = 0; i < SIZE * ITER; i++) {
        if (dst[i] != i) {
            fprintf(stderr, "Excepted %d at dst[%d], but %d\n", i, i, dst[i]);
            fflush(stderr);
            errs++;
        }
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
    space_config.num_contexts = 1;
#ifdef USE_CUDA
    space_config.memkind = SHMEMX_MEM_CUDA;
#else
    space_config.memkind = SHMEMX_MEM_HOST;
#endif
    shmemx_space_create(space_config, &space);
    shmemx_space_attach(space);

    int *src = shmemx_space_malloc(space, SIZE * ITER * sizeof(int));
    int *dst = shmemx_space_malloc(space, SIZE * ITER * sizeof(int));

    shmem_ctx_t space_ctx;
    shmemx_space_create_ctx(space, 0, &space_ctx);

    reset_data(src, dst);
    shmem_barrier_all();

    for (x = 0; x < ITER; x++) {
        int off = x * SIZE;
        if (mype == 0) {
            shmem_ctx_int_put(space_ctx, &dst[off], &src[off], SIZE, 1);
            shmem_ctx_quiet(space_ctx);
        }
    }

    shmem_barrier_all();

    if (mype == 1)
        check_data(dst);

    shmem_free(dst);
    shmem_free(src);

    shmem_ctx_destroy(space_ctx);
    shmemx_space_detach(space);
    shmemx_space_destroy(space);
    shmem_finalize();

    if (mype == 1 && errs == 0) {
        fprintf(stdout, "Passed\n");
        fflush(stderr);
    }
    return 0;
}

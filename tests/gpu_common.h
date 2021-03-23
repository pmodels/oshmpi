/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2021 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef GPU_COMMON_H_INCLUDED
#define GPU_COMMON_H_INCLUDED

#ifdef USE_CUDA
#include <cuda_runtime_api.h>

static void init_device(int mype)
{
    int dev_id = 0, dev_count = 0;

    cudaGetDeviceCount(&dev_count);
    dev_id = mype % dev_count;
    cudaSetDevice(dev_id);
    fprintf(stdout, "PE %d cudaSetDevice %d\n", mype, dev_id);
    fflush(stdout);
}
#else
static void init_device(int mype)
{
    return;
}
#endif

#endif /* GPU_COMMON_H_INCLUDED */

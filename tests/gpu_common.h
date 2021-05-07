/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2021 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef GPU_COMMON_H_INCLUDED
#define GPU_COMMON_H_INCLUDED

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#elif defined(USE_ZE)
#include <level_zero/ze_api.h>
#endif

void init_device(int mype, void **device_handle);
void reset_data(int mype, int size, int iter, int *src, int *dst);
int check_data(int size, int iter, int *dst);

#endif /* GPU_COMMON_H_INCLUDED */

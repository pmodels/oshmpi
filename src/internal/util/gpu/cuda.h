/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_UTIL_CUDA_H
#define INTERNAL_UTIL_CUDA_H

#include "oshmpi_util.h"
#ifdef OSHMPI_ENABLE_CUDA
#include <cuda_runtime_api.h>

OSHMPI_STATIC_INLINE_PREFIX OSHMPIU_gpu_pointer_type_t OSHMPIU_gpu_query_pointer_type(const void
                                                                                      *ptr)
{
    cudaError_t ret;
    struct cudaPointerAttributes ptr_attr;
    OSHMPIU_gpu_pointer_type_t type = OSHMPIU_GPU_POINTER_UNREGISTERED_HOST;

    ret = cudaPointerGetAttributes(&ptr_attr, ptr);
    if (ret == cudaSuccess) {
        switch (ptr_attr.type) {
            case cudaMemoryTypeUnregistered:
                type = OSHMPIU_GPU_POINTER_UNREGISTERED_HOST;
                break;
            case cudaMemoryTypeHost:
                type = OSHMPIU_GPU_POINTER_REGISTERED_HOST;
                break;
            case cudaMemoryTypeDevice:
                type = OSHMPIU_GPU_POINTER_DEV;
                break;
            case cudaMemoryTypeManaged:
                type = OSHMPIU_GPU_POINTER_MANAGED;
                break;
        }
    } else if (ret == cudaErrorInvalidValue) {
        type = OSHMPIU_GPU_POINTER_UNREGISTERED_HOST;
    } else
        OSHMPI_ERR_ABORT("cudaPointerGetAttributes failed, ret %d: %s\n", ret,
                         cudaGetErrorString(ret));

    return type;
}
#endif

#endif /* INTERNAL_UTIL_CUDA_H */

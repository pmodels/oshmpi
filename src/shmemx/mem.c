/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include "oshmpi_impl.h"

#ifdef OSHMPI_ENABLE_CUDA_SYMM_HEAP
void *shmemx_cuda_malloc(size_t size)
{
    return OSHMPI_cuda_malloc(size);
}

void shmemx_cuda_free(void *ptr)
{
    OSHMPI_cuda_free(ptr);
}
#endif

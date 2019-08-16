/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include "oshmpi_impl.h"

#ifdef OSHMPI_ENABLE_CUDA_SYMM_HEAP
void shmemx_cuda_symm_heap_init(void)
{
    OSHMPI_initialize_cuda_symm_heap();
}

void shmemx_cuda_symm_heap_destroy(void)
{
    OSHMPI_destroy_cuda_symm_heap();
}
#endif

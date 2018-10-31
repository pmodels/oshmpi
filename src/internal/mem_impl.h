/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#ifndef INTERNAL_MEM_IMPL_H
#define INTERNAL_MEM_IMPL_H

#include <shmem.h>
#include "oshmpi_impl.h"

OSHMPI_STATIC_INLINE_PREFIX void *OSHMPI_malloc(size_t size)
{
    void *ptr = NULL;

    ptr = mspace_malloc(OSHMPI_global.symm_heap_mspace, size);
    OSHMPI_barrier_all();
    return ptr;
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_free(void *ptr)
{
    OSHMPI_barrier_all();
    return mspace_free(OSHMPI_global.symm_heap_mspace, ptr);
}

OSHMPI_STATIC_INLINE_PREFIX void *OSHMPI_realloc(void *ptr, size_t size)
{
    void *rptr = NULL;

    rptr = mspace_realloc(OSHMPI_global.symm_heap_mspace, ptr, size);
    OSHMPI_barrier_all();
    return rptr;
}

OSHMPI_STATIC_INLINE_PREFIX void *OSHMPI_align(size_t alignment, size_t size)
{
    void *ptr = NULL;

    ptr = mspace_memalign(OSHMPI_global.symm_heap_mspace, alignment, size);
    OSHMPI_barrier_all();
    return ptr;
}

OSHMPI_STATIC_INLINE_PREFIX void *OSHMPI_calloc(size_t size)
{
    void *ptr = NULL;

    ptr = mspace_malloc(OSHMPI_global.symm_heap_mspace, size);
    OSHMPI_barrier_all();
    return ptr;
}

#endif /* INTERNAL_MEM_IMPL_H */

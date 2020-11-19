/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include "oshmpi_impl.h"

void *shmem_malloc(size_t size)
{
#pragma noinline recursive
    return OSHMPI_malloc(size);
}

void shmem_free(void *ptr)
{
#pragma noinline recursive
    OSHMPI_free(ptr);
}

void *shmem_realloc(void *ptr, size_t size)
{
#pragma noinline recursive
    return OSHMPI_realloc(ptr, size);
}

void *shmem_align(size_t alignment, size_t size)
{
#pragma noinline recursive
    return OSHMPI_align(alignment, size);
}

void *shmem_calloc(size_t count, size_t size)
{
    void *ptr = NULL;

#pragma noinline recursive
    ptr = OSHMPI_malloc(size);
    memset(ptr, 0, count * size);

    return ptr;
}

/* (deprecated APIs) */

void *shmalloc(size_t size)
{
    return shmem_malloc(size);
}

void shfree(void *ptr)
{
    return shmem_free(ptr);
}

void *shrealloc(void *ptr, size_t size)
{
    return shmem_realloc(ptr, size);
}

void *shmemalign(size_t alignment, size_t size)
{
    return shmem_align(alignment, size);
}

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

    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.symm_heap_mspace_cs);
    ptr = mspace_malloc(OSHMPI_global.symm_heap_mspace, size);
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.symm_heap_mspace_cs);

    OSHMPI_DBGMSG("size %ld, ptr %p, disp 0x%lx\n", size, ptr,
                  (MPI_Aint) ptr - (MPI_Aint) OSHMPI_global.symm_heap_base);
    OSHMPI_barrier_all();
    return ptr;
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_free(void *ptr)
{
    OSHMPI_DBGMSG("ptr %p\n", ptr);
    OSHMPI_barrier_all();

    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.symm_heap_mspace_cs);
    mspace_free(OSHMPI_global.symm_heap_mspace, ptr);
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.symm_heap_mspace_cs);
}

OSHMPI_STATIC_INLINE_PREFIX void *OSHMPI_realloc(void *ptr, size_t size)
{
    void *rptr = NULL;

    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.symm_heap_mspace_cs);
    rptr = mspace_realloc(OSHMPI_global.symm_heap_mspace, ptr, size);
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.symm_heap_mspace_cs);

    OSHMPI_DBGMSG("ptr %p size %ld -> %p\n", ptr, size, rptr);
    OSHMPI_barrier_all();
    return rptr;
}

OSHMPI_STATIC_INLINE_PREFIX void *OSHMPI_align(size_t alignment, size_t size)
{
    void *ptr = NULL;

    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.symm_heap_mspace_cs);
    ptr = mspace_memalign(OSHMPI_global.symm_heap_mspace, alignment, size);
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.symm_heap_mspace_cs);

    OSHMPI_DBGMSG("alignment %ld size %ld -> %p\n", alignment, size, ptr);
    OSHMPI_barrier_all();
    return ptr;
}

#ifdef OSHMPI_ENABLE_CUDA_SYMM_HEAP
OSHMPI_STATIC_INLINE_PREFIX void *OSHMPI_cuda_malloc(size_t size)
{
    void *ptr = NULL;

    /* TODO: A naive memory pool implementation for GPU symmetric heap.
     * Buffer cannot be reused after free. We should use a better version.*/
    ptr =
        (void *) (OSHMPI_global.cuda_symm_heap_offset + (char *) OSHMPI_global.cuda_symm_heap_base);
    OSHMPI_DBGMSG("size %ld, ptr %p, disp 0x%lx\n", size, ptr, OSHMPI_global.cuda_symm_heap_offset);

    OSHMPI_global.cuda_symm_heap_offset += size;
    OSHMPI_ASSERT(OSHMPI_global.cuda_symm_heap_offset <= OSHMPI_global.cuda_symm_heap_size);

    OSHMPI_barrier_all();
    return ptr;
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_cuda_free(void *ptr)
{
    OSHMPI_DBGMSG("ptr %p\n", ptr);
    OSHMPI_barrier_all();
}
#endif
#endif /* INTERNAL_MEM_IMPL_H */

/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include "oshmpi_impl.h"

void *OSHMPI_malloc(size_t size)
{
    void *ptr = NULL;

    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.symm_heap_mspace_cs);
    ptr = mspace_malloc(OSHMPI_global.symm_heap_mspace, size);
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.symm_heap_mspace_cs);

    OSHMPI_DBGMSG("size %ld, ptr %p, disp 0x%lx\n", size, ptr,
                  (MPI_Aint) ptr - (MPI_Aint) OSHMPI_global.symm_heap_attr.base);
    OSHMPI_barrier_all();
    return ptr;
}

void OSHMPI_free(void *ptr)
{
    OSHMPI_DBGMSG("ptr %p\n", ptr);
    OSHMPI_barrier_all();

    /* Check default symm heap */
    if (OSHMPI_sobj_check_range(ptr, OSHMPI_global.symm_heap_attr)) {
        OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.symm_heap_mspace_cs);
        mspace_free(OSHMPI_global.symm_heap_mspace, ptr);
        OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.symm_heap_mspace_cs);
    } else {
        /* Check space symm heaps */
        OSHMPI_space_t *space, *tmp;
        OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.space_list.cs);
        LL_FOREACH_SAFE(OSHMPI_global.space_list.head, space, tmp) {
            if (OSHMPI_sobj_check_range(ptr, space->sobj_attr)) {
                OSHMPI_space_free(space, ptr);
                break;
            }
        }
        OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.space_list.cs);
    }
}

void *OSHMPI_realloc(void *ptr, size_t size)
{
    void *rptr = NULL;

    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.symm_heap_mspace_cs);
    rptr = mspace_realloc(OSHMPI_global.symm_heap_mspace, ptr, size);
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.symm_heap_mspace_cs);

    OSHMPI_DBGMSG("ptr %p size %ld -> %p\n", ptr, size, rptr);
    OSHMPI_barrier_all();
    return rptr;
}

void *OSHMPI_align(size_t alignment, size_t size)
{
    void *ptr = NULL;

    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.symm_heap_mspace_cs);
    ptr = mspace_memalign(OSHMPI_global.symm_heap_mspace, alignment, size);
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.symm_heap_mspace_cs);

    OSHMPI_DBGMSG("alignment %ld size %ld -> %p\n", alignment, size, ptr);
    OSHMPI_barrier_all();
    return ptr;
}

void *OSHMPI_calloc(size_t count, size_t size)
{
    void *ptr = NULL;

    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.symm_heap_mspace_cs);
    ptr = mspace_calloc(OSHMPI_global.symm_heap_mspace, count, size);
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.symm_heap_mspace_cs);

    OSHMPI_DBGMSG("count %ld, size %ld, ptr %p, disp 0x%lx\n", count, size, ptr,
                  (MPI_Aint) ptr - (MPI_Aint) OSHMPI_global.symm_heap_attr.base);
    OSHMPI_barrier_all();
    return ptr;
}

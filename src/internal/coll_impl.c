/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#include "oshmpi_impl.h"

OSHMPI_comm_cache_t OSHMPI_coll_comm_cache;

void OSHMPI_coll_initialize(void)
{
    OSHMPI_coll_comm_cache.nobjs = 0;
    OSHMPI_coll_comm_cache.head = NULL;
    OSHMPI_THREAD_INIT_CS(&OSHMPI_coll_comm_cache.cs);

    /* FIXME: do we need preallocated cache pool allocated by the main
     * thread ? The cache object may be created by any of the threads
     * in multithreaded program. */
}

void OSHMPI_coll_finalize(void)
{
    OSHMPI_comm_cache_obj_t *cobj, *tmp;

    /* Release all cached comm */
    LL_FOREACH_SAFE(OSHMPI_coll_comm_cache.head, cobj, tmp) {
        LL_DELETE(OSHMPI_coll_comm_cache.head, cobj);
        OSHMPI_CALLMPI(MPI_Group_free(&cobj->group));
        OSHMPI_CALLMPI(MPI_Comm_free(&cobj->comm));
        OSHMPIU_free(cobj);
        OSHMPI_coll_comm_cache.nobjs--;
    }
    OSHMPI_ASSERT(OSHMPI_coll_comm_cache.nobjs == 0);
    OSHMPI_THREAD_DESTROY_CS(&OSHMPI_coll_comm_cache.cs);
}

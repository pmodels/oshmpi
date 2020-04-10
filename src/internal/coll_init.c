/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#include "oshmpi_impl.h"

#define OSHMPI_COMM_CACHE_PREALLOC 8
OSHMPI_comm_cache_obj_t OSHMPI_comm_cache_prealloc[OSHMPI_COMM_CACHE_PREALLOC];

OSHMPI_comm_cache_t OSHMPI_coll_comm_cache;

void OSHMPI_coll_initialize(void)
{
    OSHMPI_coll_comm_cache.head = NULL;
    OSHMPI_coll_comm_cache.nobjs = 0;
    memset(OSHMPI_comm_cache_prealloc, 0, sizeof(OSHMPI_comm_cache_prealloc));
    OSHMPIU_mempool_initialize(&OSHMPI_coll_comm_cache.mempool, sizeof(OSHMPI_comm_cache_obj_t),
                               (void *) OSHMPI_comm_cache_prealloc, OSHMPI_COMM_CACHE_PREALLOC);
    OSHMPI_THREAD_INIT_CS(&OSHMPI_coll_comm_cache.thread_cs);
}

void OSHMPI_coll_finalize(void)
{
    OSHMPI_comm_cache_obj_t *cobj, *tmp;

    /* Release all cached comm */
    LL_FOREACH_SAFE(OSHMPI_coll_comm_cache.head, cobj, tmp) {
        LL_DELETE(OSHMPI_coll_comm_cache.head, cobj);
        OSHMPI_CALLMPI(MPI_Group_free(&cobj->group));
        OSHMPI_CALLMPI(MPI_Comm_free(&cobj->comm));
        OSHMPIU_mempool_free_obj(&OSHMPI_coll_comm_cache.mempool, cobj);
        OSHMPI_coll_comm_cache.nobjs--;
    }
    OSHMPI_ASSERT(OSHMPI_coll_comm_cache.nobjs == 0);
    OSHMPIU_mempool_destroy(&OSHMPI_coll_comm_cache.mempool);
    OSHMPI_THREAD_DESTROY_CS(&OSHMPI_coll_comm_cache.thread_cs);
}

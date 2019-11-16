/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#include "oshmpi_impl.h"

#define OSHMPI_COMM_CACHE_PREALLOC 8
OSHMPI_comm_cache_obj_t OSHMPI_comm_cache_prealloc[OSHMPI_COMM_CACHE_PREALLOC] = { {0}
};

OSHMPI_mempool_t OSHMPI_comm_cache_mem = { NULL, 0, sizeof(OSHMPI_comm_cache_obj_t),
    NULL, 0, (void *) OSHMPI_comm_cache_prealloc, OSHMPI_COMM_CACHE_PREALLOC
};

void OSHMPI_coll_initialize(void)
{
    OSHMPI_global.comm_cache_list.nobjs = 0;
    OSHMPI_global.comm_cache_list.head = NULL;
    OSHMPI_THREAD_INIT_CS(&OSHMPI_global.comm_cache_list_cs);
}

void OSHMPI_coll_finalize(void)
{
    OSHMPI_comm_cache_obj_t *cobj, *tmp;

    /* Release all cached comm */
    LL_FOREACH_SAFE(OSHMPI_global.comm_cache_list.head, cobj, tmp) {
        LL_DELETE(OSHMPI_global.comm_cache_list.head, cobj);
        OSHMPI_CALLMPI(MPI_Group_free(&cobj->group));
        OSHMPI_CALLMPI(MPI_Comm_free(&cobj->comm));
        OSHMPIU_mempool_free_obj(&OSHMPI_comm_cache_mem, cobj);
        OSHMPI_global.comm_cache_list.nobjs--;
    }
    OSHMPI_ASSERT(OSHMPI_global.comm_cache_list.nobjs == 0);
    OSHMPIU_mempool_destroy(&OSHMPI_comm_cache_mem);
    OSHMPI_THREAD_DESTROY_CS(&OSHMPI_global.comm_cache_list_cs);
}

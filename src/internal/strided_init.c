/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include "oshmpi_impl.h"

#define OSHMPI_STRIDED_CACHE_PREALLOC 8
OSHMPI_dtype_cache_obj_t OSHMPI_strided_cache_prealloc[OSHMPI_STRIDED_CACHE_PREALLOC] = { {0}
};

OSHMPI_mempool_t OSHMPI_strided_cache_mem = { NULL, 0, sizeof(OSHMPI_dtype_cache_obj_t),
    NULL, 0, (void *) OSHMPI_strided_cache_prealloc, OSHMPI_STRIDED_CACHE_PREALLOC
};

void OSHMPI_strided_initialize(void)
{
#ifdef OSHMPI_ENABLE_STRIDED_DTYPE_CACHE
    OSHMPI_global.strided_dtype_cache.nobjs = 0;
    OSHMPI_global.strided_dtype_cache.head = NULL;
    OSHMPI_THREAD_INIT_CS(&OSHMPI_global.strided_dtype_cache_cs);
#endif
}

void OSHMPI_strided_finalize(void)
{
#ifdef OSHMPI_ENABLE_STRIDED_DTYPE_CACHE
    OSHMPI_dtype_cache_obj_t *dobj, *tmp;

    /* Release all cached datatypes */
    LL_FOREACH_SAFE(OSHMPI_global.strided_dtype_cache.head, dobj, tmp) {
        LL_DELETE(OSHMPI_global.strided_dtype_cache.head, dobj);
        OSHMPI_CALLMPI(MPI_Type_free(&dobj->sdtype));
        OSHMPIU_mempool_free_obj(&OSHMPI_strided_cache_mem, dobj);
        OSHMPI_global.strided_dtype_cache.nobjs--;
    }
    OSHMPI_ASSERT(OSHMPI_global.strided_dtype_cache.nobjs == 0);
    OSHMPIU_mempool_destroy(&OSHMPI_strided_cache_mem);
    OSHMPI_THREAD_DESTROY_CS(&OSHMPI_global.strided_dtype_cache_cs);
#endif
}

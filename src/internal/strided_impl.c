/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include "oshmpi_impl.h"

void OSHMPI_strided_initialize(void)
{
#ifdef OSHMPI_ENABLE_STRIDED_DTYPE_CACHE
    OSHMPI_global.strided_dtype_cache.nobjs = 0;
    OSHMPI_global.strided_dtype_cache.head = NULL;
    OSHMPI_THREAD_INIT_CS(&OSHMPI_global.strided_dtype_cache_cs);

    /* FIXME: do we need preallocated cache pool allocated by the main
     * thread ? The cache object may be created by any of the threads
     * in multithreaded program. */
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
        OSHMPIU_free(dobj);
        OSHMPI_global.strided_dtype_cache.nobjs--;
    }
    OSHMPI_ASSERT(OSHMPI_global.strided_dtype_cache.nobjs == 0);
    OSHMPI_THREAD_DESTROY_CS(&OSHMPI_global.strided_dtype_cache_cs);
#endif
}

/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#include <shmem.h>
#include "oshmpi_impl.h"

int shmem_ctx_create(long options, shmem_ctx_t * ctx)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
    return 0;
}

void shmem_ctx_destroy(shmem_ctx_t ctx)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#include <shmem.h>
#include "oshmpi_impl.h"

void shmem_fence(void)
{
    OSHMPI_ctx_fence(SHMEM_CTX_DEFAULT);
}

void shmem_ctx_fence(shmem_ctx_t ctx)
{
    OSHMPI_ctx_fence(ctx);
}

void shmem_quiet(void)
{
    OSHMPI_ctx_quiet(SHMEM_CTX_DEFAULT);
}

void shmem_ctx_quiet(shmem_ctx_t ctx)
{
    OSHMPI_ctx_quiet(ctx);
}

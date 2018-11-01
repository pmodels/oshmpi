/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#include <shmem.h>
#include "oshmpi_impl.h"

int shmem_ctx_create(long options, shmem_ctx_t * ctx)
{
    /* Return nonzero value if context cannot be created.
     * We cannot support it in OSHMPI unless MPI exposes the communication
     * resource management to users (e.g., endpoint).
     *
     * Spec v1.4 defines: an unsuccessful context creation call is not treated
     * as an error and the OpenSHMEM library remains in a correct state. */
    return SHMEM_NO_CTX;
}

void shmem_ctx_destroy(shmem_ctx_t ctx)
{
}

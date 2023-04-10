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
     * Spec v1.5 defines: an unsuccessful context creation call is not treated
     * as an error and the OpenSHMEM library remains in a correct state. */
    *ctx = SHMEM_CTX_INVALID;
    return SHMEM_NO_CTX;
}

void shmem_ctx_destroy(shmem_ctx_t ctx)
{
    if (ctx == SHMEM_CTX_INVALID || ctx == SHMEM_CTX_DEFAULT) {
        return;
    }
    OSHMPI_ctx_destroy((OSHMPI_ctx_t *) ctx);
}

int shmem_team_create_ctx(shmem_team_t team, long options, shmem_ctx_t * ctx)
{
    if (team == SHMEM_TEAM_INVALID) {
        *ctx = SHMEM_CTX_INVALID;
        return SHMEM_OTHER_ERR;
    }
    *ctx = SHMEM_CTX_INVALID;
    return SHMEM_NO_CTX;
}

int shmem_ctx_get_team(shmem_ctx_t ctx, shmem_team_t * team)
{
    if (ctx == SHMEM_CTX_DEFAULT) {
        *team = SHMEM_TEAM_WORLD;
        return SHMEM_SUCCESS;
    }
    *team = SHMEM_TEAM_INVALID;
    return SHMEM_OTHER_ERR;
}

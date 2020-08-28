/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#include <shmem.h>
#include "oshmpi_impl.h"

void OSHMPI_ctx_destroy(OSHMPI_ctx_t * ctx)
{
    if (ctx == SHMEM_CTX_DEFAULT)
        return;

    OSHMPI_ATOMIC_FLAG_STORE(ctx->used_flag, 0);

    /* Do not free window.
     * Because the only possible context is created with space, relying
     * on collective attach/detach to create/free the internal window.
     * All contexts will be freed at detach.
     * Same approach will be used for team based context creation.*/
}

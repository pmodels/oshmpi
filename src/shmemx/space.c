/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include <shmemx.h>
#include "oshmpi_impl.h"

void shmemx_space_create(shmemx_space_config_t space_config, shmemx_space_t * space)
{
    OSHMPI_NOINLINE_RECURSIVE()
        OSHMPI_space_create(space_config, (OSHMPI_space_t **) space);
}

void shmemx_space_destroy(shmemx_space_t space)
{
    OSHMPI_NOINLINE_RECURSIVE()
        OSHMPI_space_destroy(space);
}

int shmemx_space_create_ctx(shmemx_space_t space, long options, shmem_ctx_t * ctx)
{
    OSHMPI_NOINLINE_RECURSIVE()
        return OSHMPI_space_create_ctx((OSHMPI_space_t *) space, options, (OSHMPI_ctx_t **) ctx);
}

void shmemx_space_attach(shmemx_space_t space)
{
    OSHMPI_NOINLINE_RECURSIVE()
        OSHMPI_space_attach((OSHMPI_space_t *) space);
}

void shmemx_space_detach(shmemx_space_t space)
{
    OSHMPI_NOINLINE_RECURSIVE()
        OSHMPI_space_detach((OSHMPI_space_t *) space);
}

void *shmemx_space_malloc(shmemx_space_t space, size_t size)
{
    void *ptr = NULL;
    OSHMPI_NOINLINE_RECURSIVE()
        ptr = OSHMPI_space_malloc((OSHMPI_space_t *) space, size);
    return ptr;
}

void *shmemx_space_calloc(shmemx_space_t space, size_t count, size_t size)
{
    void *ptr = NULL;
    OSHMPI_NOINLINE_RECURSIVE()
        ptr = OSHMPI_space_malloc((OSHMPI_space_t *) space, size);
    memset(ptr, 0, size);
    return ptr;
}

void *shmemx_space_align(shmemx_space_t space, size_t alignment, size_t size)
{
    void *ptr = NULL;
    OSHMPI_NOINLINE_RECURSIVE()
        ptr = OSHMPI_space_align((OSHMPI_space_t *) space, alignment, size);
    return ptr;
}

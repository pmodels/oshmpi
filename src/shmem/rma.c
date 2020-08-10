/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include "oshmpi_impl.h"

void shmem_ctx_putmem(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
{
    OSHMPI_ctx_put(ctx, MPI_BYTE, 1, source, dest, nelems, pe);
}

void shmem_putmem(void *dest, const void *source, size_t nelems, int pe)
{
    OSHMPI_ctx_put(SHMEM_CTX_DEFAULT, MPI_BYTE, 1, source, dest, nelems, pe);
}

void shmem_ctx_getmem(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
{
    OSHMPI_ctx_get(ctx, MPI_BYTE, 1, dest, source, nelems, pe);
}

void shmem_getmem(void *dest, const void *source, size_t nelems, int pe)
{
    OSHMPI_ctx_get(SHMEM_CTX_DEFAULT, MPI_BYTE, 1, dest, source, nelems, pe);
}

void shmem_ctx_putmem_nbi(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
{
    OSHMPI_ctx_put_nbi(ctx, MPI_BYTE, 1, source /* origin_addr */ , dest /* target_addr */ , nelems,
                       pe);
}

void shmem_putmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{
    OSHMPI_ctx_put_nbi(SHMEM_CTX_DEFAULT, MPI_BYTE, 1, source /* origin_addr */ ,
                       dest /* target_addr */ , nelems, pe);
}

void shmem_ctx_getmem_nbi(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
{
    OSHMPI_ctx_get_nbi(ctx, MPI_BYTE, 1, dest /* origin_addr */ , source /* target_addr */ , nelems,
                       pe);
}

void shmem_getmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{
    OSHMPI_ctx_get_nbi(SHMEM_CTX_DEFAULT, MPI_BYTE, 1, dest /* origin_addr */ ,
                       source /* target_addr */ , nelems, pe);
}

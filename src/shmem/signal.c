/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2022 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include "shmem.h"
#include "oshmpi_impl.h"

void shmem_putmem_signal(void *dest, const void *source, size_t nelems, uint64_t * sig_addr,
                         uint64_t signal, int sig_op, int pe)
{
    OSHMPI_ASSERT(0);
}

void shmem_ctx_putmem_signal(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems,
                             uint64_t * sig_addr, uint64_t signal, int sig_op, int pe)
{
    OSHMPI_ASSERT(0);
}

void shmem_putmem_signal_nbi(void *dest, const void *source, size_t nelems, uint64_t * sig_addr,
                             uint64_t signal, int sig_op, int pe)
{
    OSHMPI_ASSERT(0);
}

void shmem_ctx_putmem_signal_nbi(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems,
                                 uint64_t * sig_addr, uint64_t signal, int sig_op, int pe)
{
    OSHMPI_ASSERT(0);
}

void shmem_signal_fetch(const uint64_t * sig_addr)
{
    OSHMPI_ASSERT(0);
}

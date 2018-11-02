/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_ORDER_IMPL_H
#define INTERNAL_ORDER_IMPL_H

#include "oshmpi_impl.h"

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_fence(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)))
{
    /* Ensure ordered delivery of all outstanding Put, AMO, and nonblocking Put */
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_heap_win));
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_data_win));

    /* Ensure ordered delivery of all outstanding AM AMOs */
    OSHMPI_amo_am_flush_all(ctx);

    /* Ensure ordered delivery of memory store */
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_heap_win));
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_data_win));
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_quiet(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)))
{
    /* Ensure completion of all outstanding Put, AMO, nonblocking Put and Get */
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_heap_win));
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_data_win));

    /* Ensure completion of all outstanding AM AMOs */
    OSHMPI_amo_am_flush_all(ctx);

    /* Ensure completion of memory store */
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_heap_win));
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_data_win));
}

#endif /* INTERNAL_ORDER_IMPL_H */

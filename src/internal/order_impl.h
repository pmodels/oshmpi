/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_ORDER_IMPL_H
#define INTERNAL_ORDER_IMPL_H

#include "oshmpi_impl.h"

#ifdef OSHMPI_ENABLE_OP_TRACKING
#define CHECK_FLAG(flag) (flag)
#else
#define CHECK_FLAG(flag) (1)
#endif

#ifdef OSHMPI_ENABLE_OP_TRACKING
#define RESET_FLAG(flag) do {flag = 0;} while (0)
#else
#define RESET_FLAG(flag) do {} while (0)
#endif

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_fence(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)))
{
    if (CHECK_FLAG(OSHMPI_global.symm_heap_outstanding_op)) {
        /* Ensure ordered delivery of all outstanding Put, AMO, and nonblocking Put */
        OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_heap_win));
        /* Ensure ordered delivery of memory store */
        OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_heap_win));

        OSHMPI_DBGMSG("fence: flushed symm heap.\n");
        RESET_FLAG(OSHMPI_global.symm_heap_outstanding_op);
    }

    if (CHECK_FLAG(OSHMPI_global.symm_data_outstanding_op)) {
        /* Ensure ordered delivery of all outstanding Put, AMO, and nonblocking Put */
        OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_data_win));
        /* Ensure ordered delivery of memory store */
        OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_data_win));

        OSHMPI_DBGMSG("fence: flushed symm data.\n");
        RESET_FLAG(OSHMPI_global.symm_data_outstanding_op);
    }

#ifdef OSHMPI_ENABLE_CUDA_SYMM_HEAP
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.cuda_symm_heap_win));
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.cuda_symm_heap_win));
#endif

    /* Ensure special AMO ordered delivery (e.g., AM AMOs) */
    OSHMPI_amo_flush_all(ctx);
}

OSHMPI_TIMER_EXTERN_DECL(quiet);

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_quiet(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)))
{
    OSHMPI_TIMER_LOCAL_DECL(quiet);
    OSHMPI_TIMER_START(quiet);
    if (CHECK_FLAG(OSHMPI_global.symm_heap_outstanding_op)) {
        /* Ensure completion of all outstanding Put, AMO, nonblocking Put and Get */
        OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_heap_win));
        /* Ensure completion of memory store */
        OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_heap_win));

        OSHMPI_DBGMSG("quiet: flushed symm heap.\n");
        RESET_FLAG(OSHMPI_global.symm_heap_outstanding_op);
    }

    if (CHECK_FLAG(OSHMPI_global.symm_data_outstanding_op)) {
        /* Ensure completion of all outstanding Put, AMO, nonblocking Put and Get */
        OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_data_win));
        /* Ensure completion of memory store */
        OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_data_win));

        OSHMPI_DBGMSG("quiet: flushed symm data.\n");
        RESET_FLAG(OSHMPI_global.symm_data_outstanding_op);
    }

#ifdef OSHMPI_ENABLE_CUDA_SYMM_HEAP
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.cuda_symm_heap_win));
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.cuda_symm_heap_win));
#endif

    /* Ensure special AMO ordered delivery (e.g., AM AMOs) */
    OSHMPI_amo_flush_all(ctx);
    OSHMPI_TIMER_START(quiet);
}

#endif /* INTERNAL_ORDER_IMPL_H */

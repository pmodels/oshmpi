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

OSHMPI_STATIC_INLINE_PREFIX void ctx_flush_impl(OSHMPI_ictx_t * ictx)
{
    if (CHECK_FLAG(ictx->outstanding_op)) {
        /* Ensure ordered delivery of all outstanding Put, AMO, and nonblocking Put */
        OSHMPI_FORCEINLINE()
            OSHMPI_CALLMPI(MPI_Win_flush_all(ictx->win));
        /* Ensure ordered delivery of memory store */
        OSHMPI_FORCEINLINE()
            OSHMPI_CALLMPI(MPI_Win_sync(ictx->win));
        RESET_FLAG(ictx->outstanding_op);
    }
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_fence(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)))
{
    if (ctx != SHMEM_CTX_DEFAULT) {
        /* Fence on specific context (window) */
        ctx_flush_impl(&(((OSHMPI_ctx_t *) ctx)->ictx));
        return;
    }
#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
    ctx_flush_impl(&OSHMPI_global.symm_ictx);
#else
    ctx_flush_impl(&OSHMPI_global.symm_heap_ictx);
    ctx_flush_impl(&OSHMPI_global.symm_data_ictx);
#endif

    /* Flush all space contexts */
    OSHMPI_space_t *space, *tmp;
    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.space_list.cs);
    LL_FOREACH_SAFE(OSHMPI_global.space_list.head, space, tmp) {
        int i;
        for (i = 0; i < space->config.num_contexts; i++)
            ctx_flush_impl(&(space->ctx_list[i].ictx));
        ctx_flush_impl(&space->default_ictx);
    }
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.space_list.cs);

    /* Ensure special AMO ordered delivery (e.g., AM AMOs) */
    OSHMPI_amo_flush_all(ctx);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_quiet(shmem_ctx_t ctx)
{
    if (ctx != SHMEM_CTX_DEFAULT) {
        /* Quiet on specific context (window) */
        ctx_flush_impl(&(((OSHMPI_ctx_t *) ctx)->ictx));
        return;
    }
#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
    ctx_flush_impl(&OSHMPI_global.symm_ictx);
#else
    ctx_flush_impl(&OSHMPI_global.symm_heap_ictx);
    ctx_flush_impl(&OSHMPI_global.symm_data_ictx);
#endif

    /* Flush all space contexts */
    OSHMPI_space_t *space, *tmp;
    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.space_list.cs);
    LL_FOREACH_SAFE(OSHMPI_global.space_list.head, space, tmp) {
        int i;
        for (i = 0; i < space->config.num_contexts; i++)
            ctx_flush_impl(&(space->ctx_list[i].ictx));
        ctx_flush_impl(&space->default_ictx);
    }
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.space_list.cs);

    /* Ensure special AMO ordered delivery (e.g., AM AMOs) */
    OSHMPI_amo_flush_all(ctx);
}

#endif /* INTERNAL_ORDER_IMPL_H */

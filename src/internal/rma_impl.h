/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#ifndef INTERNAL_RMA_IMPL_H
#define INTERNAL_RMA_IMPL_H

#include "oshmpi_impl.h"

OSHMPI_STATIC_INLINE_PREFIX void ctx_put_nbi_impl(OSHMPI_ctx_t * ctx,
                                                  MPI_Datatype origin_type,
                                                  MPI_Datatype target_type, const void *origin_addr,
                                                  void *target_addr, size_t origin_count,
                                                  size_t target_count, int pe,
                                                  OSHMPI_ictx_t ** ictx_ptr)
{
    MPI_Aint target_disp = -1;
    OSHMPI_ictx_t *ictx = NULL;

    OSHMPI_translate_ictx_disp(ctx, (const void *) target_addr, pe, &target_disp, &ictx,
                               NULL /*sobj_handle_ptr */);
    OSHMPI_ASSERT(target_disp >= 0 && ictx);

    /* TODO: check non-int inputs exceeds int limit */

    OSHMPI_FORCEINLINE()
        OSHMPI_CALLMPI(MPI_Put(origin_addr, (int) origin_count, origin_type, pe,
                               target_disp, (int) target_count, target_type, ictx->win));
    OSHMPI_SET_OUTSTANDING_OP(ictx, OSHMPI_OP_OUTSTANDING);     /* PUT is always outstanding */

    OSHMPI_DBGMSG("dest %p, disp 0x%lx, win=0x%lx\n", target_addr, target_disp, (uint64_t) ictx->win);

    /* return context object if the caller requires */
    if (ictx_ptr != NULL)
        *ictx_ptr = ictx;
}

OSHMPI_STATIC_INLINE_PREFIX void ctx_get_nbi_impl(OSHMPI_ctx_t * ctx,
                                                  MPI_Datatype origin_type,
                                                  MPI_Datatype target_type, void *origin_addr,
                                                  const void *target_addr, size_t origin_count,
                                                  size_t target_count, int pe, int completion,
                                                  OSHMPI_ictx_t ** ictx_ptr)
{
    MPI_Aint target_disp = -1;
    OSHMPI_ictx_t *ictx = NULL;

    OSHMPI_translate_ictx_disp(ctx, (const void *) target_addr, pe, &target_disp, &ictx,
                               NULL /*sobj_handle_ptr */);
    OSHMPI_ASSERT(target_disp >= 0 && ictx);

    /* TODO: check non-int inputs exceeds int limit */
    OSHMPI_FORCEINLINE()
        OSHMPI_CALLMPI(MPI_Get(origin_addr, (int) origin_count, origin_type, pe,
                               target_disp, (int) target_count, target_type, ictx->win));
    OSHMPI_SET_OUTSTANDING_OP(ictx, completion);        /* GET can be outstanding or completed */

    /* return context object if the caller requires */
    if (ictx_ptr != NULL)
        *ictx_ptr = ictx;
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_put_nbi(shmem_ctx_t ctx,
                                                    MPI_Datatype mpi_type, const void *origin_addr,
                                                    void *target_addr, size_t nelems, int pe)
{
    if (nelems == 0)
        return;

    ctx_put_nbi_impl((OSHMPI_ctx_t *) ctx, mpi_type, mpi_type, origin_addr, target_addr, nelems,
                     nelems, pe, NULL);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_put(shmem_ctx_t ctx,
                                                MPI_Datatype mpi_type, const void *origin_addr,
                                                void *target_addr, size_t nelems, int pe)
{
    OSHMPI_ictx_t *ictx = NULL;

    if (nelems == 0)
        return;

    ctx_put_nbi_impl((OSHMPI_ctx_t *) ctx, mpi_type, mpi_type, origin_addr, target_addr, nelems,
                     nelems, pe, &ictx);
    ctx_local_complete_impl(pe, ictx);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_iput(shmem_ctx_t ctx,
                                                 MPI_Datatype mpi_type, const void *origin_addr,
                                                 void *target_addr, ptrdiff_t target_st,
                                                 ptrdiff_t origin_st, size_t nelems, int pe)
{
    OSHMPI_ictx_t *ictx = NULL;
    MPI_Datatype origin_type = MPI_DATATYPE_NULL, target_type = MPI_DATATYPE_NULL;
    size_t origin_count = 0, target_count = 0;

    if (nelems == 0)
        return;

    OSHMPI_create_strided_dtype(nelems, origin_st, mpi_type, 0 /* no required extent */ ,
                                &origin_count, &origin_type);
    if (origin_st == target_st) {
        target_type = origin_type;
        target_count = origin_count;
    } else
        OSHMPI_create_strided_dtype(nelems, target_st, mpi_type, 0 /* no required extent */ ,
                                    &target_count, &target_type);

    ctx_put_nbi_impl((OSHMPI_ctx_t *) ctx, origin_type, target_type, origin_addr, target_addr,
                     origin_count, target_count, pe, &ictx);
    ctx_local_complete_impl(pe, ictx);

    OSHMPI_free_strided_dtype(mpi_type, &origin_type);
    if (origin_st != target_st)
        OSHMPI_free_strided_dtype(mpi_type, &target_type);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_get_nbi(shmem_ctx_t ctx,
                                                    MPI_Datatype mpi_type, void *origin_addr,
                                                    const void *target_addr, size_t nelems, int pe)
{
    if (nelems == 0)
        return;

    /* TODO: check non-int inputs exceeds int limit */
    ctx_get_nbi_impl((OSHMPI_ctx_t *) ctx, mpi_type, mpi_type, origin_addr, target_addr,
                     nelems, nelems, pe, OSHMPI_OP_OUTSTANDING, NULL);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_get(shmem_ctx_t ctx,
                                                MPI_Datatype mpi_type, void *origin_addr,
                                                const void *target_addr, size_t nelems, int pe)
{
    OSHMPI_ictx_t *ictx = NULL;

    if (nelems == 0)
        return;

    /* TODO: check non-int inputs exceeds int limit */
    ctx_get_nbi_impl((OSHMPI_ctx_t *) ctx, mpi_type, mpi_type, origin_addr, target_addr,
                     nelems, nelems, pe, OSHMPI_OP_COMPLETED, &ictx);
    ctx_local_complete_impl(pe, ictx);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_iget(shmem_ctx_t ctx,
                                                 MPI_Datatype mpi_type, void *origin_addr,
                                                 const void *target_addr, ptrdiff_t origin_st,
                                                 ptrdiff_t target_st, size_t nelems, int pe)
{
    OSHMPI_ictx_t *ictx = NULL;
    MPI_Datatype origin_type = MPI_DATATYPE_NULL, target_type = MPI_DATATYPE_NULL;
    size_t origin_count = 0, target_count = 0;

    if (nelems == 0)
        return;

    OSHMPI_create_strided_dtype(nelems, origin_st, mpi_type, 0 /* no required extent */ ,
                                &origin_count, &origin_type);
    if (origin_st == target_st) {
        target_type = origin_type;
        target_count = origin_count;
    } else
        OSHMPI_create_strided_dtype(nelems, target_st, mpi_type, 0 /* no required extent */ ,
                                    &target_count, &target_type);

    ctx_get_nbi_impl((OSHMPI_ctx_t *) ctx, origin_type, target_type, origin_addr, target_addr,
                     origin_count, target_count, pe, OSHMPI_OP_COMPLETED, &ictx);
    ctx_local_complete_impl(pe, ictx);

    OSHMPI_free_strided_dtype(mpi_type, &origin_type);
    if (origin_st != target_st)
        OSHMPI_free_strided_dtype(mpi_type, &target_type);
}

#endif /* INTERNAL_RMA_IMPL_H */

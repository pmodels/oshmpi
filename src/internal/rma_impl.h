/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#ifndef INTERNAL_RMA_IMPL_H
#define INTERNAL_RMA_IMPL_H

#include "oshmpi_impl.h"
#include "rma_am_impl.h"

OSHMPI_STATIC_INLINE_PREFIX void ctx_put_nbi_impl(OSHMPI_ictx_t * ictx,
                                                  MPI_Datatype origin_type,
                                                  MPI_Datatype target_type, const void *origin_addr,
                                                  void *target_addr, size_t origin_count,
                                                  size_t target_count, int pe,
                                                  OSHMPI_sobj_attr_t * sobj_attr)
{
    MPI_Aint target_disp = -1;

    /* TODO: check non-int inputs exceeds int limit */

    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, (const void *) target_addr, pe,
                                    OSHMPI_ICTX_DISP_MODE(ictx), &target_disp);
    OSHMPI_ASSERT(target_disp >= 0);

    OSHMPI_FORCEINLINE()
        OSHMPI_CALLMPI(MPI_Put(origin_addr, (int) origin_count, origin_type, pe,
                               target_disp, (int) target_count, target_type, ictx->win));
    OSHMPI_SET_OUTSTANDING_OP(ictx, OSHMPI_OP_OUTSTANDING);     /* PUT is always outstanding */
}

OSHMPI_STATIC_INLINE_PREFIX void ctx_get_nbi_impl(OSHMPI_ictx_t * ictx,
                                                  MPI_Datatype origin_type,
                                                  MPI_Datatype target_type, void *origin_addr,
                                                  const void *target_addr, size_t origin_count,
                                                  size_t target_count, int pe, int completion,
                                                  OSHMPI_sobj_attr_t * sobj_attr)
{
    MPI_Aint target_disp = -1;

    /* TODO: check non-int inputs exceeds int limit */

    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, target_addr, pe,
                                    OSHMPI_ICTX_DISP_MODE(ictx), &target_disp);
    OSHMPI_ASSERT(target_disp >= 0);

    OSHMPI_FORCEINLINE()
        OSHMPI_CALLMPI(MPI_Get(origin_addr, (int) origin_count, origin_type, pe,
                               target_disp, (int) target_count, target_type, ictx->win));
    OSHMPI_SET_OUTSTANDING_OP(ictx, completion);        /* GET can be outstanding or completed */
}

OSHMPI_STATIC_INLINE_PREFIX void ctx_iput_nbi_impl(OSHMPI_ictx_t * ictx,
                                                   MPI_Datatype mpi_type, const void *origin_addr,
                                                   void *target_addr, ptrdiff_t target_st,
                                                   ptrdiff_t origin_st, size_t nelems, int pe,
                                                   OSHMPI_sobj_attr_t * sobj_attr)
{
    MPI_Datatype origin_type = MPI_DATATYPE_NULL, target_type = MPI_DATATYPE_NULL;
    size_t origin_count = 0, target_count = 0;

    OSHMPI_create_strided_dtype(nelems, origin_st, mpi_type, 0 /* no required extent */ ,
                                &origin_count, &origin_type);
    if (origin_st == target_st) {
        target_type = origin_type;
        target_count = origin_count;
    } else
        OSHMPI_create_strided_dtype(nelems, target_st, mpi_type, 0 /* no required extent */ ,
                                    &target_count, &target_type);

    ctx_put_nbi_impl(ictx, origin_type, target_type, origin_addr, target_addr,
                     origin_count, target_count, pe, sobj_attr);

    OSHMPI_free_strided_dtype(mpi_type, &origin_type);
    if (origin_st != target_st)
        OSHMPI_free_strided_dtype(mpi_type, &target_type);
}

OSHMPI_STATIC_INLINE_PREFIX void ctx_iget_nbi_impl(OSHMPI_ictx_t * ictx,
                                                   MPI_Datatype mpi_type, void *origin_addr,
                                                   const void *target_addr, ptrdiff_t origin_st,
                                                   ptrdiff_t target_st, size_t nelems, int pe,
                                                   int completion, OSHMPI_sobj_attr_t * sobj_attr)
{
    MPI_Datatype origin_type = MPI_DATATYPE_NULL, target_type = MPI_DATATYPE_NULL;
    size_t origin_count = 0, target_count = 0;

    OSHMPI_create_strided_dtype(nelems, origin_st, mpi_type, 0 /* no required extent */ ,
                                &origin_count, &origin_type);
    if (origin_st == target_st) {
        target_type = origin_type;
        target_count = origin_count;
    } else
        OSHMPI_create_strided_dtype(nelems, target_st, mpi_type, 0 /* no required extent */ ,
                                    &target_count, &target_type);

    ctx_get_nbi_impl(ictx, origin_type, target_type, origin_addr, target_addr,
                     origin_count, target_count, pe, completion, sobj_attr);

    OSHMPI_free_strided_dtype(mpi_type, &origin_type);
    if (origin_st != target_st)
        OSHMPI_free_strided_dtype(mpi_type, &target_type);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_put_nbi(shmem_ctx_t ctx,
                                                    MPI_Datatype mpi_type, size_t typesz,
                                                    const void *origin_addr, void *target_addr,
                                                    size_t nelems, int pe)
{
    OSHMPI_ictx_t *ictx = NULL;
    OSHMPI_sobj_attr_t *sobj_attr = NULL;

    if (nelems == 0)
        return;

    OSHMPI_sobj_query_attr_ictx(ctx, (const void *) target_addr, pe, &sobj_attr, &ictx);
    OSHMPI_ASSERT(sobj_attr && ictx);

    if (OSHMPI_ENABLE_DIRECT_RMA_RUNTIME(origin_addr, sobj_attr->memkind, OSHMPI_PUT))
        ctx_put_nbi_impl(ictx, mpi_type, mpi_type, origin_addr, target_addr, nelems,
                         nelems, pe, sobj_attr);
    else        /* TODO: nonblocking version of AM PUT */
        OSHMPI_rma_am_put(ictx, mpi_type, typesz, origin_addr, target_addr, nelems, pe, sobj_attr);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_put(shmem_ctx_t ctx,
                                                MPI_Datatype mpi_type, size_t typesz,
                                                const void *origin_addr, void *target_addr,
                                                size_t nelems, int pe)
{
    OSHMPI_ictx_t *ictx = NULL;
    OSHMPI_sobj_attr_t *sobj_attr = NULL;

    if (nelems == 0)
        return;

    OSHMPI_sobj_query_attr_ictx(ctx, (const void *) target_addr, pe, &sobj_attr, &ictx);
    OSHMPI_ASSERT(sobj_attr && ictx);

    if (OSHMPI_ENABLE_DIRECT_RMA_RUNTIME(origin_addr, sobj_attr->memkind, OSHMPI_PUT)) {
        ctx_put_nbi_impl(ictx, mpi_type, mpi_type, origin_addr, target_addr, nelems,
                         nelems, pe, sobj_attr);
        ctx_local_complete_impl(pe, ictx);
    } else
        OSHMPI_rma_am_put(ictx, mpi_type, typesz, origin_addr, target_addr, nelems, pe, sobj_attr);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_iput(shmem_ctx_t ctx,
                                                 MPI_Datatype mpi_type,
                                                 OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                 const void *origin_addr, void *target_addr,
                                                 ptrdiff_t target_st, ptrdiff_t origin_st,
                                                 size_t nelems, int pe)
{
    OSHMPI_ictx_t *ictx = NULL;
    OSHMPI_sobj_attr_t *sobj_attr = NULL;

    if (nelems == 0)
        return;

    OSHMPI_sobj_query_attr_ictx(ctx, (const void *) target_addr, pe, &sobj_attr, &ictx);
    OSHMPI_ASSERT(sobj_attr && ictx);

    if (OSHMPI_ENABLE_DIRECT_RMA_RUNTIME(origin_addr, sobj_attr->memkind, OSHMPI_PUT)) {
        ctx_iput_nbi_impl(ictx, mpi_type, origin_addr, target_addr,
                          target_st, origin_st, nelems, pe, sobj_attr);
        ctx_local_complete_impl(pe, ictx);
    } else
        OSHMPI_rma_am_iput(ictx, mpi_type, mpi_type_idx, origin_addr, target_addr,
                           origin_st, target_st, nelems, pe, sobj_attr);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_get_nbi(shmem_ctx_t ctx,
                                                    MPI_Datatype mpi_type, size_t typesz,
                                                    void *origin_addr, const void *target_addr,
                                                    size_t nelems, int pe)
{
    OSHMPI_ictx_t *ictx = NULL;
    OSHMPI_sobj_attr_t *sobj_attr = NULL;

    if (nelems == 0)
        return;

    OSHMPI_sobj_query_attr_ictx(ctx, target_addr, pe, &sobj_attr, &ictx);
    OSHMPI_ASSERT(sobj_attr && ictx);

    if (OSHMPI_ENABLE_DIRECT_RMA_RUNTIME(origin_addr, sobj_attr->memkind, OSHMPI_GET)) {
        /* TODO: check non-int inputs exceeds int limit */
        ctx_get_nbi_impl(ictx, mpi_type, mpi_type, origin_addr, target_addr,
                         nelems, nelems, pe, OSHMPI_OP_OUTSTANDING, sobj_attr);
    } else      /* TODO: nonblocking version of AM GET */
        OSHMPI_rma_am_get(ictx, mpi_type, typesz, origin_addr, target_addr, nelems, pe, sobj_attr);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_get(shmem_ctx_t ctx,
                                                MPI_Datatype mpi_type, size_t typesz,
                                                void *origin_addr, const void *target_addr,
                                                size_t nelems, int pe)
{
    OSHMPI_ictx_t *ictx = NULL;
    OSHMPI_sobj_attr_t *sobj_attr = NULL;

    if (nelems == 0)
        return;

    OSHMPI_sobj_query_attr_ictx(ctx, target_addr, pe, &sobj_attr, &ictx);
    OSHMPI_ASSERT(sobj_attr && ictx);

    if (OSHMPI_ENABLE_DIRECT_RMA_RUNTIME(origin_addr, sobj_attr->memkind, OSHMPI_GET)) {
        /* TODO: check non-int inputs exceeds int limit */
        ctx_get_nbi_impl(ictx, mpi_type, mpi_type, origin_addr, target_addr,
                         nelems, nelems, pe, OSHMPI_OP_COMPLETED, sobj_attr);
        ctx_local_complete_impl(pe, ictx);
    } else
        OSHMPI_rma_am_get(ictx, mpi_type, typesz, origin_addr, target_addr, nelems, pe, sobj_attr);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_iget(shmem_ctx_t ctx,
                                                 MPI_Datatype mpi_type,
                                                 OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                 void *origin_addr, const void *target_addr,
                                                 ptrdiff_t origin_st, ptrdiff_t target_st,
                                                 size_t nelems, int pe)
{
    OSHMPI_ictx_t *ictx = NULL;
    OSHMPI_sobj_attr_t *sobj_attr = NULL;

    if (nelems == 0)
        return;

    OSHMPI_sobj_query_attr_ictx(ctx, target_addr, pe, &sobj_attr, &ictx);
    OSHMPI_ASSERT(sobj_attr && ictx);

    if (OSHMPI_ENABLE_DIRECT_RMA_RUNTIME(origin_addr, sobj_attr->memkind, OSHMPI_GET)) {
        ctx_iget_nbi_impl(ictx, mpi_type, origin_addr, target_addr,
                          origin_st, target_st, nelems, pe, OSHMPI_OP_COMPLETED, sobj_attr);
        ctx_local_complete_impl(pe, ictx);
    } else
        OSHMPI_rma_am_iget(ictx, mpi_type, mpi_type_idx, origin_addr, target_addr,
                           origin_st, target_st, nelems, pe, sobj_attr);
}

#endif /* INTERNAL_RMA_IMPL_H */

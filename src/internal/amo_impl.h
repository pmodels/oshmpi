/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_AMO_IMPL_H
#define INTERNAL_AMO_IMPL_H

#include "oshmpi_impl.h"
#include "amo_am_impl.h"

OSHMPI_STATIC_INLINE_PREFIX void ctx_fetch_op_impl(shmem_ctx_t ctx,
                                                   MPI_Datatype mpi_type, const void *origin_addr,
                                                   void *result_addr, void *target_addr, MPI_Op op,
                                                   int pe)
{
    MPI_Aint target_disp = -1;
    OSHMPI_ictx_t *ictx = NULL;

    OSHMPI_translate_ictx_disp(ctx, (const void *) target_addr, pe, &target_disp, &ictx,
                               NULL /* sobj_attr_ptr */);
    OSHMPI_ASSERT(target_disp >= 0 && ictx);

    OSHMPI_CALLMPI(MPI_Fetch_and_op
                   (origin_addr, result_addr, mpi_type, pe, target_disp, op, ictx->win));

    ctx_local_complete_impl(pe, ictx);

    if (op == MPI_NO_OP)
        OSHMPI_SET_OUTSTANDING_OP(ictx, OSHMPI_OP_COMPLETED);   /* FETCH-only is always completed */
    else
        OSHMPI_SET_OUTSTANDING_OP(ictx, OSHMPI_OP_OUTSTANDING);
}

OSHMPI_STATIC_INLINE_PREFIX void ctx_set_op_impl(shmem_ctx_t ctx,
                                                 MPI_Datatype mpi_type, const void *origin_addr,
                                                 void *target_addr, MPI_Op op, int pe)
{
    MPI_Aint target_disp = -1;
    OSHMPI_ictx_t *ictx = NULL;

    OSHMPI_translate_ictx_disp(ctx, (const void *) target_addr, pe, &target_disp, &ictx,
                               NULL /* sobj_attr_ptr */);
    OSHMPI_ASSERT(target_disp >= 0 && ictx);

    OSHMPI_CALLMPI(MPI_Accumulate
                   (origin_addr, 1, mpi_type, pe, target_disp, 1, mpi_type, op, ictx->win));

    ctx_local_complete_impl(pe, ictx);

    OSHMPI_SET_OUTSTANDING_OP(ictx, OSHMPI_OP_OUTSTANDING);     /* SET is always outstanding */
}

OSHMPI_STATIC_INLINE_PREFIX void ctx_cswap_impl(shmem_ctx_t ctx,
                                                MPI_Datatype mpi_type,
                                                void *target_addr,
                                                void *compare_addr,
                                                void *origin_addr, int pe, void *result_addr)
{
    MPI_Aint target_disp = -1;
    OSHMPI_ictx_t *ictx = NULL;

    OSHMPI_translate_ictx_disp(ctx, (const void *) target_addr, pe, &target_disp, &ictx,
                               NULL /* sobj_attr_ptr */);
    OSHMPI_ASSERT(target_disp >= 0 && ictx);

    OSHMPI_CALLMPI(MPI_Compare_and_swap
                   (origin_addr, compare_addr, result_addr, mpi_type, pe, target_disp, ictx->win));

    ctx_local_complete_impl(pe, ictx);

    OSHMPI_SET_OUTSTANDING_OP(ictx, OSHMPI_OP_OUTSTANDING);     /* remote SWAP is outstanding */
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_cswap(shmem_ctx_t ctx
                                                  OSHMPI_ATTRIBUTE((unused)), MPI_Datatype mpi_type,
                                                  OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                  size_t bytes, void *dest /* target_addr */ ,
                                                  void *cond_ptr /*compare_addr */ ,
                                                  void *value_ptr /* origin_addr */ ,
                                                  int pe, void *oldval_ptr /*result_addr */)
{
    if (OSHMPI_ENABLE_DIRECT_AMO_RUNTIME)
        ctx_cswap_impl(ctx, mpi_type, dest, cond_ptr, value_ptr, pe, oldval_ptr);
    else
        OSHMPI_amo_am_cswap(ctx, mpi_type, mpi_type_idx, bytes, dest, cond_ptr,
                            value_ptr, pe, oldval_ptr);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_fetch(shmem_ctx_t ctx
                                                  OSHMPI_ATTRIBUTE((unused)), MPI_Datatype mpi_type,
                                                  OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                  size_t bytes, MPI_Op op,
                                                  OSHMPI_am_mpi_op_index_t op_idx,
                                                  void *dest /* target_addr */ ,
                                                  void *value_ptr /* origin_addr */ ,
                                                  int pe, void *oldval_ptr /* result_addr */)
{
    if (OSHMPI_ENABLE_DIRECT_AMO_RUNTIME)
        ctx_fetch_op_impl(ctx, mpi_type, value_ptr, oldval_ptr, dest, op, pe);
    else
        OSHMPI_amo_am_fetch(ctx, mpi_type, mpi_type_idx, bytes, op, op_idx, dest,
                            value_ptr, pe, oldval_ptr);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_post(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 MPI_Datatype mpi_type,
                                                 OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                 size_t bytes, MPI_Op op,
                                                 OSHMPI_am_mpi_op_index_t op_idx,
                                                 void *dest /* target_addr */ ,
                                                 void *value_ptr /* origin_addr */ ,
                                                 int pe)
{
    if (OSHMPI_ENABLE_DIRECT_AMO_RUNTIME)
        ctx_set_op_impl(ctx, mpi_type, value_ptr, dest, op, pe);
    else
        OSHMPI_amo_am_post(ctx, mpi_type, mpi_type_idx, bytes, op, op_idx, dest, value_ptr, pe);
}

#endif /* INTERNAL_AMO_IMPL_H */

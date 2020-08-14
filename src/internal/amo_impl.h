/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_AMO_IMPL_H
#define INTERNAL_AMO_IMPL_H

#include "oshmpi_impl.h"
#include "amo_am_impl.h"

OSHMPI_STATIC_INLINE_PREFIX void ctx_fetch_op_impl(OSHMPI_ictx_t * ictx,
                                                   MPI_Datatype mpi_type, const void *origin_addr,
                                                   void *result_addr, void *target_addr, MPI_Op op,
                                                   int pe, OSHMPI_sobj_attr_t * sobj_attr)
{
    MPI_Aint target_disp = -1;

    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, target_addr, pe,
                                    OSHMPI_ICTX_DISP_MODE(ictx), &target_disp);
    OSHMPI_ASSERT(target_disp >= 0);

    OSHMPI_CALLMPI(MPI_Fetch_and_op
                   (origin_addr, result_addr, mpi_type, pe, target_disp, op, ictx->win));

    ctx_local_complete_impl(pe, ictx);

    if (op == MPI_NO_OP)
        OSHMPI_SET_OUTSTANDING_OP(ictx, OSHMPI_OP_COMPLETED);   /* FETCH-only is always completed */
    else
        OSHMPI_SET_OUTSTANDING_OP(ictx, OSHMPI_OP_OUTSTANDING);
}

OSHMPI_STATIC_INLINE_PREFIX void ctx_set_op_impl(OSHMPI_ictx_t * ictx,
                                                 MPI_Datatype mpi_type, const void *origin_addr,
                                                 void *target_addr, MPI_Op op, int pe,
                                                 OSHMPI_sobj_attr_t * sobj_attr)
{
    MPI_Aint target_disp = -1;

    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, target_addr, pe,
                                    OSHMPI_ICTX_DISP_MODE(ictx), &target_disp);
    OSHMPI_ASSERT(target_disp >= 0);

    OSHMPI_CALLMPI(MPI_Accumulate
                   (origin_addr, 1, mpi_type, pe, target_disp, 1, mpi_type, op, ictx->win));

    ctx_local_complete_impl(pe, ictx);

    OSHMPI_SET_OUTSTANDING_OP(ictx, OSHMPI_OP_OUTSTANDING);     /* SET is always outstanding */
}

OSHMPI_STATIC_INLINE_PREFIX void ctx_cswap_impl(OSHMPI_ictx_t * ictx,
                                                MPI_Datatype mpi_type,
                                                void *target_addr,
                                                void *compare_addr,
                                                void *origin_addr, int pe, void *result_addr,
                                                OSHMPI_sobj_attr_t * sobj_attr)
{
    MPI_Aint target_disp = -1;

    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, target_addr, pe,
                                    OSHMPI_ICTX_DISP_MODE(ictx), &target_disp);
    OSHMPI_ASSERT(target_disp >= 0);

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
    OSHMPI_sobj_attr_t *sobj_attr = NULL;
    OSHMPI_ictx_t *ictx = NULL;

    OSHMPI_sobj_query_attr_ictx(ctx, (const void *) dest, pe, &sobj_attr, &ictx);
    OSHMPI_ASSERT(sobj_attr && ictx);

    if (OSHMPI_ENABLE_DIRECT_AMO_RUNTIME)
        ctx_cswap_impl(ictx, mpi_type, dest, cond_ptr, value_ptr, pe, oldval_ptr, sobj_attr);
    else
        OSHMPI_amo_am_cswap(ictx, mpi_type, mpi_type_idx, bytes, dest, cond_ptr,
                            value_ptr, pe, oldval_ptr, sobj_attr);
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
    OSHMPI_sobj_attr_t *sobj_attr = NULL;
    OSHMPI_ictx_t *ictx = NULL;

    OSHMPI_sobj_query_attr_ictx(ctx, (const void *) dest, pe, &sobj_attr, &ictx);
    OSHMPI_ASSERT(sobj_attr && ictx);

    if (OSHMPI_ENABLE_DIRECT_AMO_RUNTIME)
        ctx_fetch_op_impl(ictx, mpi_type, value_ptr, oldval_ptr, dest, op, pe, sobj_attr);
    else
        OSHMPI_amo_am_fetch(ictx, mpi_type, mpi_type_idx, bytes, op, op_idx, dest,
                            value_ptr, pe, oldval_ptr, sobj_attr);
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
    OSHMPI_sobj_attr_t *sobj_attr = NULL;
    OSHMPI_ictx_t *ictx = NULL;

    OSHMPI_sobj_query_attr_ictx(ctx, (const void *) dest, pe, &sobj_attr, &ictx);
    OSHMPI_ASSERT(sobj_attr && ictx);

    if (OSHMPI_ENABLE_DIRECT_AMO_RUNTIME)
        ctx_set_op_impl(ictx, mpi_type, value_ptr, dest, op, pe, sobj_attr);
    else
        OSHMPI_amo_am_post(ictx, mpi_type, mpi_type_idx, bytes, op, op_idx, dest,
                           value_ptr, pe, sobj_attr);
}

#endif /* INTERNAL_AMO_IMPL_H */

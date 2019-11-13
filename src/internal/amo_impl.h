/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_AMO_IMPL_H
#define INTERNAL_AMO_IMPL_H

#include "oshmpi_impl.h"

/* Default make decision at runtime */
#if !defined(OSHMPI_ENABLE_DIRECT_AMO) && !defined(OSHMPI_ENABLE_AM_AMO)

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_cswap(shmem_ctx_t ctx
                                                  OSHMPI_ATTRIBUTE((unused)), MPI_Datatype mpi_type,
                                                  OSHMPI_amo_mpi_datatype_index_t mpi_type_idx,
                                                  size_t bytes, void *dest /* target_addr */ ,
                                                  void *cond_ptr /*compare_addr */ ,
                                                  void *value_ptr /* origin_addr */ ,
                                                  int pe, void *oldval_ptr /*result_addr */)
{
    if (OSHMPI_global.amo_direct)
        OSHMPI_amo_direct_cswap(ctx, mpi_type, mpi_type_idx, bytes, dest, cond_ptr,
                                value_ptr, pe, oldval_ptr);
    else
        OSHMPI_amo_am_cswap(ctx, mpi_type, mpi_type_idx, bytes, dest, cond_ptr,
                            value_ptr, pe, oldval_ptr);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_fetch(shmem_ctx_t ctx
                                                  OSHMPI_ATTRIBUTE((unused)), MPI_Datatype mpi_type,
                                                  OSHMPI_amo_mpi_datatype_index_t mpi_type_idx,
                                                  size_t bytes, MPI_Op op,
                                                  OSHMPI_amo_mpi_op_index_t op_idx,
                                                  void *dest /* target_addr */ ,
                                                  void *value_ptr /* origin_addr */ ,
                                                  int pe, void *oldval_ptr /* result_addr */)
{
    if (OSHMPI_global.amo_direct)
        OSHMPI_amo_direct_fetch(ctx, mpi_type, mpi_type_idx, bytes, op, op_idx, dest,
                                value_ptr, pe, oldval_ptr);
    else
        OSHMPI_amo_am_fetch(ctx, mpi_type, mpi_type_idx, bytes, op, op_idx, dest,
                            value_ptr, pe, oldval_ptr);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_post(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 MPI_Datatype mpi_type,
                                                 OSHMPI_amo_mpi_datatype_index_t mpi_type_idx,
                                                 size_t bytes, MPI_Op op,
                                                 OSHMPI_amo_mpi_op_index_t op_idx,
                                                 void *dest /* target_addr */ ,
                                                 void *value_ptr /* origin_addr */ ,
                                                 int pe)
{
    if (OSHMPI_global.amo_direct)
        OSHMPI_amo_direct_post(ctx, mpi_type, mpi_type_idx, bytes, op, op_idx, dest, value_ptr, pe);
    else
        OSHMPI_amo_am_post(ctx, mpi_type, mpi_type_idx, bytes, op, op_idx, dest, value_ptr, pe);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_flush(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                  int PE_start, int logPE_stride, int PE_size)
{
    /* No separate flush is needed in direct AMO. */
    if (!OSHMPI_global.amo_direct)
        OSHMPI_amo_am_flush(ctx, PE_start, logPE_stride, PE_size);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_flush_all(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)))
{
    /* No separate flush is needed in direct AMO. */
    if (!OSHMPI_global.amo_direct)
        OSHMPI_amo_am_flush_all(ctx);
}
#endif /* !defined(OSHMPI_ENABLE_DIRECT_AMO) && !defined(OSHMPI_ENABLE_AM_AMO) */

#endif /* INTERNAL_AMO_IMPL_H */

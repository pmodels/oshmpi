/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_AMO_DIRECT_IMPL_H
#define INTERNAL_AMO_DIRECT_IMPL_H

#include "oshmpi_impl.h"

OSHMPI_STATIC_INLINE_PREFIX void ctx_fetch_op_impl(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                   MPI_Datatype mpi_type, const void *origin_addr,
                                                   void *result_addr, void *target_addr, MPI_Op op,
                                                   int pe)
{
    MPI_Aint target_disp = -1;
    MPI_Win win = MPI_WIN_NULL;

    OSHMPI_translate_win_and_disp((const void *) target_addr, &win, &target_disp);
    OSHMPI_ASSERT(target_disp >= 0 && win != MPI_WIN_NULL);

    OSHMPI_CALLMPI(MPI_Fetch_and_op(origin_addr, result_addr, mpi_type, pe, target_disp, op, win));

    ctx_local_complete_impl(ctx, pe, win);

    OSHMPI_SET_OUTSTANDING_OP(win, OSHMPI_OP_COMPLETED);        /* FETCH is always completed */
}

OSHMPI_STATIC_INLINE_PREFIX void ctx_set_op_impl(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 MPI_Datatype mpi_type, const void *origin_addr,
                                                 void *target_addr, MPI_Op op, int pe)
{
    MPI_Aint target_disp = -1;
    MPI_Win win = MPI_WIN_NULL;

    OSHMPI_translate_win_and_disp((const void *) target_addr, &win, &target_disp);
    OSHMPI_ASSERT(target_disp >= 0 && win != MPI_WIN_NULL);

    OSHMPI_CALLMPI(MPI_Accumulate(origin_addr, 1, mpi_type, pe, target_disp, 1, mpi_type, op, win));

    ctx_local_complete_impl(ctx, pe, win);

    OSHMPI_SET_OUTSTANDING_OP(win, OSHMPI_OP_OUTSTANDING);      /* SET is always outstanding */
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_direct_initialize(void)
{
    OSHMPI_DBGMSG("Initialized direct AMO\n");
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_direct_finalize(void)
{
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_direct_cswap(shmem_ctx_t ctx
                                                         OSHMPI_ATTRIBUTE((unused)),
                                                         MPI_Datatype mpi_type,
                                                         OSHMPI_amo_mpi_datatype_index_t
                                                         mpi_type_idx OSHMPI_ATTRIBUTE((unused)),
                                                         size_t bytes OSHMPI_ATTRIBUTE((unused)),
                                                         void *dest /* target_addr */ ,
                                                         void *cond_ptr /*compare_addr */ ,
                                                         void *value_ptr /* origin_addr */ ,
                                                         int pe, void *oldval_ptr /*result_addr */)
{
    MPI_Aint target_disp = -1;
    MPI_Win win = MPI_WIN_NULL;

    OSHMPI_translate_win_and_disp((const void *) dest, &win, &target_disp);
    OSHMPI_ASSERT(target_disp >= 0 && win != MPI_WIN_NULL);

    OSHMPI_CALLMPI(MPI_Compare_and_swap
                   (value_ptr, cond_ptr, oldval_ptr, mpi_type, pe, target_disp, win));

    ctx_local_complete_impl(ctx, pe, win);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_direct_fetch(shmem_ctx_t ctx
                                                         OSHMPI_ATTRIBUTE((unused)),
                                                         MPI_Datatype mpi_type,
                                                         OSHMPI_amo_mpi_datatype_index_t
                                                         mpi_type_idx OSHMPI_ATTRIBUTE((unused)),
                                                         size_t bytes OSHMPI_ATTRIBUTE((unused)),
                                                         MPI_Op op,
                                                         OSHMPI_amo_mpi_op_index_t op_idx
                                                         OSHMPI_ATTRIBUTE((unused)),
                                                         void *dest /* target_addr */ ,
                                                         void *value_ptr /* origin_addr */ ,
                                                         int pe,
                                                         void *oldval_ptr /* result_addr */)
{
    ctx_fetch_op_impl(ctx, mpi_type, value_ptr, oldval_ptr, dest, op, pe);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_direct_post(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                        MPI_Datatype mpi_type,
                                                        OSHMPI_amo_mpi_datatype_index_t mpi_type_idx
                                                        OSHMPI_ATTRIBUTE((unused)),
                                                        size_t bytes OSHMPI_ATTRIBUTE((unused)),
                                                        MPI_Op op,
                                                        OSHMPI_amo_mpi_op_index_t op_idx
                                                        OSHMPI_ATTRIBUTE((unused)),
                                                        void *dest /* target_addr */ ,
                                                        void *value_ptr /* origin_addr */ ,
                                                        int pe)
{
    ctx_set_op_impl(ctx, mpi_type, value_ptr, dest, op, pe);
}

#ifdef OSHMPI_ENABLE_DIRECT_AMO
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_initialize(void)
{
    OSHMPI_amo_direct_initialize();
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_finalize(void)
{
    OSHMPI_amo_direct_finalize();
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_cb_progress(void)
{
    /* No callback progress needed in direct AMO. */
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_cswap(shmem_ctx_t ctx
                                                  OSHMPI_ATTRIBUTE((unused)), MPI_Datatype mpi_type,
                                                  OSHMPI_amo_mpi_datatype_index_t mpi_type_idx,
                                                  size_t bytes, void *dest /* target_addr */ ,
                                                  void *cond_ptr /*compare_addr */ ,
                                                  void *value_ptr /* origin_addr */ ,
                                                  int pe, void *oldval_ptr /*result_addr */)
{
    OSHMPI_amo_direct_cswap(ctx, mpi_type, mpi_type_idx, bytes, dest, cond_ptr,
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
    OSHMPI_amo_direct_fetch(ctx, mpi_type, mpi_type_idx, bytes, op, op_idx, dest,
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
    OSHMPI_amo_direct_post(ctx, mpi_type, mpi_type_idx, bytes, op, op_idx, dest, value_ptr, pe);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_flush(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                  int PE_start, int logPE_stride, int PE_size)
{
    /* No separate flush is needed in direct AMO. */
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_flush_all(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)))
{
    /* No separate flush is needed in direct AMO. */
}
#endif /* OSHMPI_ENABLE_DIRECT_AMO */
#endif /* INTERNAL_AMO_DIRECT_IMPL_H */

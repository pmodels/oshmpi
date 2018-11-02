/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_AMO_IMPL_H
#define INTERNAL_AMO_IMPL_H

#include "oshmpi_impl.h"

#ifdef OSHMPI_ENABLE_DIRECT_AMO
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
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_initialize(void)
{
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_finalize(void)
{
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_cb_progress(void)
{
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_cswap(shmem_ctx_t ctx
                                                  OSHMPI_ATTRIBUTE((unused)), MPI_Datatype mpi_type,
                                                  OSHMPI_amo_mpi_datatype_index_t mpi_type_idx
                                                  OSHMPI_ATTRIBUTE((unused)),
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

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_fetch(shmem_ctx_t ctx
                                                  OSHMPI_ATTRIBUTE((unused)), MPI_Datatype mpi_type,
                                                  OSHMPI_amo_mpi_datatype_index_t mpi_type_idx
                                                  OSHMPI_ATTRIBUTE((unused)),
                                                  size_t bytes OSHMPI_ATTRIBUTE((unused)),
                                                  MPI_Op op,
                                                  OSHMPI_amo_mpi_op_index_t op_idx
                                                  OSHMPI_ATTRIBUTE((unused)),
                                                  void *dest /* target_addr */ ,
                                                  void *value_ptr /* origin_addr */ ,
                                                  int pe, void *oldval_ptr /* result_addr */)
{
    ctx_fetch_op_impl(ctx, mpi_type, value_ptr, oldval_ptr, dest, op, pe);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_post(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 MPI_Datatype mpi_type,
                                                 OSHMPI_amo_mpi_datatype_index_t mpi_type_idx
                                                 OSHMPI_ATTRIBUTE((unused)),
                                                 size_t bytes OSHMPI_ATTRIBUTE((unused)), MPI_Op op,
                                                 OSHMPI_amo_mpi_op_index_t op_idx
                                                 OSHMPI_ATTRIBUTE((unused)),
                                                 void *dest /* target_addr */ ,
                                                 void *value_ptr /* origin_addr */ ,
                                                 int pe)
{
    ctx_set_op_impl(ctx, mpi_type, value_ptr, dest, op, pe);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_flush(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                     int PE_start, int logPE_stride, int PE_size)
{

}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_flush_all(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)))
{
}
#else
#include "amo_am_impl.h"
#endif
#endif /* INTERNAL_AMO_IMPL_H */

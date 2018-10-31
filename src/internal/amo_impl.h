/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_AMO_IMPL_H
#define INTERNAL_AMO_IMPL_H

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

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_compare_swap(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                         MPI_Datatype mpi_type,
                                                         const void *origin_addr,
                                                         const void *compare_addr,
                                                         void *result_addr, void *target_addr,
                                                         int pe)
{
    MPI_Aint target_disp = -1;
    MPI_Win win = MPI_WIN_NULL;

    OSHMPI_translate_win_and_disp((const void *) target_addr, &win, &target_disp);
    OSHMPI_ASSERT(target_disp >= 0 && win != MPI_WIN_NULL);

    OSHMPI_CALLMPI(MPI_Compare_and_swap
                   (origin_addr, compare_addr, result_addr, mpi_type, pe, target_disp, win));

    ctx_local_complete_impl(ctx, pe, win);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_fetch_add(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                      MPI_Datatype mpi_type,
                                                      const void *origin_addr, void *result_addr,
                                                      void *target_addr, int pe)
{
    return ctx_fetch_op_impl(ctx, mpi_type, origin_addr, result_addr, target_addr, MPI_SUM, pe);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_add(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                MPI_Datatype mpi_type, const void *origin_addr,
                                                void *target_addr, int pe)
{
    return ctx_set_op_impl(ctx, mpi_type, origin_addr, target_addr, MPI_SUM, pe);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_fetch(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                  MPI_Datatype mpi_type, void *result_addr,
                                                  const void *target_addr, int pe)
{
    return ctx_fetch_op_impl(ctx, mpi_type, NULL, result_addr, (void *) target_addr, MPI_NO_OP, pe);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_set(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                MPI_Datatype mpi_type, const void *origin_addr,
                                                void *target_addr, int pe)
{
    return ctx_set_op_impl(ctx, mpi_type, origin_addr, target_addr, MPI_REPLACE, pe);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_swap(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 MPI_Datatype mpi_type, const void *origin_addr,
                                                 void *result_addr, void *target_addr, int pe)
{
    return ctx_fetch_op_impl(ctx, mpi_type, origin_addr, result_addr, target_addr, MPI_REPLACE, pe);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_fetch_and(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                      MPI_Datatype mpi_type,
                                                      const void *origin_addr, void *result_addr,
                                                      void *target_addr, int pe)
{
    return ctx_fetch_op_impl(ctx, mpi_type, origin_addr, result_addr, target_addr, MPI_BAND, pe);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_and(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                MPI_Datatype mpi_type, const void *origin_addr,
                                                void *target_addr, int pe)
{
    return ctx_set_op_impl(ctx, mpi_type, origin_addr, target_addr, MPI_BAND, pe);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_fetch_or(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                     MPI_Datatype mpi_type, const void *origin_addr,
                                                     void *result_addr, void *target_addr, int pe)
{
    return ctx_fetch_op_impl(ctx, mpi_type, origin_addr, result_addr, target_addr, MPI_BOR, pe);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_or(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                               MPI_Datatype mpi_type, const void *origin_addr,
                                               void *target_addr, int pe)
{
    return ctx_set_op_impl(ctx, mpi_type, origin_addr, target_addr, MPI_BOR, pe);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_fetch_xor(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                      MPI_Datatype mpi_type,
                                                      const void *origin_addr, void *result_addr,
                                                      void *target_addr, int pe)
{
    return ctx_fetch_op_impl(ctx, mpi_type, origin_addr, result_addr, target_addr, MPI_BXOR, pe);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_xor(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                MPI_Datatype mpi_type, const void *origin_addr,
                                                void *target_addr, int pe)
{
    return ctx_set_op_impl(ctx, mpi_type, origin_addr, target_addr, MPI_BXOR, pe);
}
#endif /* INTERNAL_AMO_IMPL_H */

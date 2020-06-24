/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#ifndef INTERNAL_RMA_IMPL_H
#define INTERNAL_RMA_IMPL_H

#include "oshmpi_impl.h"

OSHMPI_STATIC_INLINE_PREFIX void ctx_put_nbi_impl(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                  MPI_Datatype origin_type,
                                                  MPI_Datatype target_type, const void *origin_addr,
                                                  void *target_addr, size_t origin_count,
                                                  size_t target_count, int pe, MPI_Win * win_ptr)
{
    MPI_Aint target_disp = -1;
    MPI_Win win = MPI_WIN_NULL;

    OSHMPI_translate_win_and_disp((const void *) target_addr, pe, &win, &target_disp);
    OSHMPI_ASSERT(target_disp >= 0 && win != MPI_WIN_NULL);

    /* TODO: check non-int inputs exceeds int limit */

    OSHMPI_FORCEINLINE()
        OSHMPI_CALLMPI(MPI_Put(origin_addr, (int) origin_count, origin_type, pe,
                               target_disp, (int) target_count, target_type, win));
    OSHMPI_SET_OUTSTANDING_OP(win, OSHMPI_OP_OUTSTANDING);      /* PUT is always outstanding */

    /* return window object if the caller requires */
    if (win_ptr != NULL)
        *win_ptr = win;
}

OSHMPI_STATIC_INLINE_PREFIX void ctx_get_nbi_impl(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                  MPI_Datatype origin_type,
                                                  MPI_Datatype target_type, void *origin_addr,
                                                  const void *target_addr, size_t origin_count,
                                                  size_t target_count, int pe, int completion,
                                                  MPI_Win * win_ptr)
{
    MPI_Aint target_disp = -1;
    MPI_Win win = MPI_WIN_NULL;

    OSHMPI_translate_win_and_disp((const void *) target_addr, pe, &win, &target_disp);
    OSHMPI_ASSERT(target_disp >= 0 && win != MPI_WIN_NULL);

    /* TODO: check non-int inputs exceeds int limit */

    OSHMPI_FORCEINLINE()
        OSHMPI_CALLMPI(MPI_Get(origin_addr, (int) origin_count, origin_type, pe,
                               target_disp, (int) target_count, target_type, win));
    OSHMPI_SET_OUTSTANDING_OP(win, completion); /* GET can be outstanding or completed */

    /* return window object if the caller requires */
    if (win_ptr != NULL)
        *win_ptr = win;
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_put_nbi(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                    MPI_Datatype mpi_type, const void *origin_addr,
                                                    void *target_addr, size_t nelems, int pe)
{
    if (nelems == 0)
        return;

    ctx_put_nbi_impl(ctx, mpi_type, mpi_type, origin_addr, target_addr, nelems, nelems, pe, NULL);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_put(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                MPI_Datatype mpi_type, const void *origin_addr,
                                                void *target_addr, size_t nelems, int pe)
{
    MPI_Win win = MPI_WIN_NULL;

    if (nelems == 0)
        return;

    ctx_put_nbi_impl(ctx, mpi_type, mpi_type, origin_addr, target_addr, nelems, nelems, pe, &win);
    ctx_local_complete_impl(ctx, pe, win);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_iput(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 MPI_Datatype mpi_type, const void *origin_addr,
                                                 void *target_addr, ptrdiff_t target_st,
                                                 ptrdiff_t origin_st, size_t nelems, int pe)
{
    MPI_Win win = MPI_WIN_NULL;
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

    ctx_put_nbi_impl(ctx, origin_type, target_type, origin_addr, target_addr,
                     origin_count, target_count, pe, &win);
    ctx_local_complete_impl(ctx, pe, win);

    OSHMPI_free_strided_dtype(mpi_type, &origin_type);
    if (origin_st != target_st)
        OSHMPI_free_strided_dtype(mpi_type, &target_type);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_get_nbi(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                    MPI_Datatype mpi_type, void *origin_addr,
                                                    const void *target_addr, size_t nelems, int pe)
{
    if (nelems == 0)
        return;

    /* TODO: check non-int inputs exceeds int limit */
    ctx_get_nbi_impl(ctx, mpi_type, mpi_type, origin_addr, target_addr,
                     nelems, nelems, pe, OSHMPI_OP_OUTSTANDING, NULL);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_get(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                MPI_Datatype mpi_type, void *origin_addr,
                                                const void *target_addr, size_t nelems, int pe)
{
    MPI_Win win = MPI_WIN_NULL;

    if (nelems == 0)
        return;

    /* TODO: check non-int inputs exceeds int limit */
    ctx_get_nbi_impl(ctx, mpi_type, mpi_type, origin_addr, target_addr,
                     nelems, nelems, pe, OSHMPI_OP_COMPLETED, &win);
    ctx_local_complete_impl(ctx, pe, win);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_iget(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 MPI_Datatype mpi_type, void *origin_addr,
                                                 const void *target_addr, ptrdiff_t origin_st,
                                                 ptrdiff_t target_st, size_t nelems, int pe)
{
    MPI_Win win = MPI_WIN_NULL;
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

    ctx_get_nbi_impl(ctx, origin_type, target_type, origin_addr, target_addr,
                     origin_count, target_count, pe, OSHMPI_OP_COMPLETED, &win);
    ctx_local_complete_impl(ctx, pe, win);

    OSHMPI_free_strided_dtype(mpi_type, &origin_type);
    if (origin_st != target_st)
        OSHMPI_free_strided_dtype(mpi_type, &target_type);
}

#endif /* INTERNAL_RMA_IMPL_H */

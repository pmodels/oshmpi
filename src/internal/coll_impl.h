/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_COLL_IMPL_H
#define INTERNAL_COLL_IMPL_H

/* Block until all PEs arrive at the barrier and all local updates
 * and remote memory updates on the default context are completed. */
#include "oshmpi_util.h"
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_barrier_all(void)
{
    /* Ensure completion of all outstanding Put, AMO, and nonblocking Put */
#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_ictx.win));
#else
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_heap_ictx.win));
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_data_ictx.win));
#endif
    /* Ensure AM completion (e.g., AM AMOs) */
    OSHMPI_am_flush_all(SHMEM_CTX_DEFAULT);

    /* Ensure completion of memory store */
#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_ictx.win));
#else
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_heap_ictx.win));
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_data_ictx.win));
#endif
    OSHMPI_am_progress_mpi_barrier(OSHMPI_global.team_world_comm);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_sync_all(void)
{
    /* Ensure completion of previously issued memory store */
#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_ictx.win));
#else
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_heap_ictx.win));
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_data_ictx.win));
#endif
    OSHMPI_am_progress_mpi_barrier(OSHMPI_global.team_world_comm);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_sync_team(OSHMPI_team_t * team)
{
    OSHMPI_am_progress_mpi_barrier(team->comm);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_broadcast_team(OSHMPI_team_t * team, void *dest,
                                                       const void *source, size_t nelems,
                                                       MPI_Datatype mpi_type, int PE_root)
{
    OSHMPI_am_progress_mpi_bcast(PE_root == OSHMPI_global.team_world_my_pe ? (void *) source : dest,
                                 nelems, mpi_type, PE_root, team->comm);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_collect_team(OSHMPI_team_t * team, void *dest,
                                                     const void *source, size_t nelems,
                                                     MPI_Datatype mpi_type)
{
    int *rcounts, *rdispls;
    int comm_size = 0;
    unsigned int same_nelems = 0;

    /* collect allows each PE to have different nelems. */
    OSHMPI_CALLMPI(MPI_Comm_size(team->comm, &comm_size));
    rcounts = OSHMPIU_malloc(comm_size * sizeof(int));
    OSHMPI_ASSERT(rcounts);

    rdispls = OSHMPIU_malloc(comm_size * sizeof(int));
    OSHMPI_ASSERT(rdispls);

    OSHMPI_am_progress_mpi_allgather(&nelems, 1, MPI_INT, rcounts, 1, MPI_INT, team->comm);

    rdispls[0] = 0;
    same_nelems = (nelems == rcounts[0]) ? 1 : 0;
    for (int i = 1; i < comm_size; i++) {
        rdispls[i] = rdispls[i - 1] + rcounts[i - 1];
        same_nelems &= (nelems == rcounts[i]);
    }

    if (same_nelems)    /* call faster allgather if same nelems on all PEs */
        OSHMPI_am_progress_mpi_allgather(source, nelems, mpi_type, dest, nelems, mpi_type,
                                         team->comm);
    else
        OSHMPI_am_progress_mpi_allgatherv(source, nelems, mpi_type, dest, rcounts, rdispls,
                                          mpi_type, team->comm);

    OSHMPIU_free(rdispls);
    OSHMPIU_free(rcounts);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_fcollect_team(OSHMPI_team_t * team, void *dest,
                                                      const void *source, size_t nelems,
                                                      MPI_Datatype mpi_type)
{
    OSHMPI_am_progress_mpi_allgather(source, nelems, mpi_type, dest, nelems, mpi_type, team->comm);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_alltoall_team(OSHMPI_team_t * team, void *dest,
                                                      const void *source, size_t nelems,
                                                      MPI_Datatype mpi_type)
{
    OSHMPI_am_progress_mpi_alltoall(source, nelems, mpi_type, dest, nelems, mpi_type, team->comm);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_alltoalls_team(OSHMPI_team_t * team, void *dest,
                                                       const void *source, ptrdiff_t dst,
                                                       ptrdiff_t sst, size_t nelems,
                                                       MPI_Datatype mpi_type)
{
    MPI_Datatype sdtype = MPI_DATATYPE_NULL, rdtype = MPI_DATATYPE_NULL;
    size_t scount, rcount;

    /* Values of dst, sst, nelems must be equal on all PEs. When dst=sst=1, same as alltoall.
     * TODO: not sure if alltoall with ddt is faster or alltoallv is faster */

    /* Create derived datatypes if strided > 1, otherwise directly use basic datatype;
     * when dst == sst, reuse send datatype. */
    OSHMPI_create_strided_dtype(nelems, sst, mpi_type, nelems * sst /* required extent */ ,
                                &scount, &sdtype);
    if (dst == sst) {
        rdtype = sdtype;
        rcount = scount;
    } else
        OSHMPI_create_strided_dtype(nelems, dst, mpi_type, nelems * dst /* required extent */ ,
                                    &rcount, &rdtype);

    /* TODO: check non-int inputs exceeds int limit */


    OSHMPI_am_progress_mpi_alltoall(source, (int) scount, sdtype, dest, (int) rcount, rdtype,
                                    team->comm);

    OSHMPI_free_strided_dtype(mpi_type, &sdtype);
    if (dst != sst)
        OSHMPI_free_strided_dtype(mpi_type, &rdtype);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_allreduce_team(OSHMPI_team_t * team, void *dest,
                                                       const void *source, size_t nreduce,
                                                       MPI_Datatype mpi_type, MPI_Op op)
{
    /* source and dest may be the same array, but may not be overlapping. */
    OSHMPI_am_progress_mpi_allreduce((source == dest) ? MPI_IN_PLACE : source,
                                     dest, (int) nreduce, mpi_type, op, team->comm);
}

#endif /* INTERNAL_COLL_IMPL_H */

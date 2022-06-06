/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#include <shmem.h>
#include "oshmpi_impl.h"

void shmem_barrier_all(void)
{
    OSHMPI_barrier_all();
}

int shmem_team_sync(shmem_team_t team)
{
    OSHMPI_ASSERT(0);
    return SHMEM_OTHER_ERR;
}

void shmem_sync_all(void)
{
    OSHMPI_sync_all();
}

int shmem_broadcastmem(shmem_team_t team, void *dest, const void *source, size_t nelems,
                       int PE_root)
{
    OSHMPI_team_t *team_obj;
    OSHMPI_TEAM_GET_OBJ(team, team_obj);
    OSHMPI_broadcast_team(team_obj, dest, source, nelems, OSHMPI_MPI_COLL_BYTE_T, PE_root);
    return SHMEM_SUCCESS;
}

int shmem_collectmem(shmem_team_t team, void *dest, const void *source, size_t nelems)
{
    OSHMPI_team_t *team_obj;
    OSHMPI_TEAM_GET_OBJ(team, team_obj);
    OSHMPI_collect_team(team_obj, dest, source, nelems, OSHMPI_MPI_COLL_BYTE_T);
    return SHMEM_SUCCESS;
}

int shmem_fcollectmem(shmem_team_t team, void *dest, const void *source, size_t nelems)
{
    OSHMPI_team_t *team_obj;
    OSHMPI_TEAM_GET_OBJ(team, team_obj);
    OSHMPI_fcollect_team(team_obj, dest, source, nelems, OSHMPI_MPI_COLL_BYTE_T);
    return SHMEM_SUCCESS;
}

int shmem_alltoallmem(shmem_team_t team, void *dest, const void *source, size_t nelems)
{
    OSHMPI_team_t *team_obj;
    OSHMPI_TEAM_GET_OBJ(team, team_obj);
    OSHMPI_alltoall_team(team_obj, dest, source, nelems, OSHMPI_MPI_COLL_BYTE_T);
    return SHMEM_SUCCESS;
}

int shmem_alltoallsmem(shmem_team_t team, void *dest, const void *source, ptrdiff_t dst,
                       ptrdiff_t sst, size_t nelems)
{
    OSHMPI_team_t *team_obj;
    OSHMPI_TEAM_GET_OBJ(team, team_obj);
    OSHMPI_alltoalls_team(team_obj, dest, source, dst, sst, nelems, OSHMPI_MPI_COLL_BYTE_T);
    return SHMEM_SUCCESS;
}

/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2022 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#include <shmem.h>
#include "oshmpi_impl.h"

/* TPL_BLOCK_START */

int shmem_TYPENAME_broadcast(shmem_team_t team, TYPE * dest, const TYPE * source, size_t nelems,
                             int PE_root)
{
    OSHMPI_team_t *team_obj;
    OSHMPI_TEAM_GET_OBJ(team, team_obj);
    OSHMPI_broadcast_team(team_obj, dest, source, nelems, MPI_TYPE, PE_root);
    return SHMEM_SUCCESS;
}
int shmem_TYPENAME_collect(shmem_team_t team, TYPE * dest, const TYPE * source, size_t nelems)
{
    OSHMPI_team_t *team_obj;
    OSHMPI_TEAM_GET_OBJ(team, team_obj);
    OSHMPI_collect_team(team_obj, dest, source, nelems, MPI_TYPE);
    return SHMEM_SUCCESS;
}
int shmem_TYPENAME_fcollect(shmem_team_t team, TYPE * dest, const TYPE * source, size_t nelems)
{
    OSHMPI_team_t *team_obj;
    OSHMPI_TEAM_GET_OBJ(team, team_obj);
    OSHMPI_fcollect_team(team_obj, dest, source, nelems, MPI_TYPE);
    return SHMEM_SUCCESS;
}

int shmem_TYPENAME_alltoall(shmem_team_t team, TYPE * dest, const TYPE * source, size_t nelems)
{
    OSHMPI_team_t *team_obj;
    OSHMPI_TEAM_GET_OBJ(team, team_obj);
    OSHMPI_alltoall_team(team_obj, dest, source, nelems, MPI_TYPE);
    return SHMEM_SUCCESS;
}

int shmem_TYPENAME_alltoalls(shmem_team_t team, TYPE * dest, const TYPE * source, ptrdiff_t dst,
                             ptrdiff_t sst, size_t nelems)
{
    OSHMPI_team_t *team_obj;
    OSHMPI_TEAM_GET_OBJ(team, team_obj);
    OSHMPI_alltoalls_team(team_obj, dest, source, dst, sst, nelems, MPI_TYPE);
    return SHMEM_SUCCESS;
}
/* TPL_BLOCK_END */

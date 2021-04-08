/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2021 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#include <shmem.h>
#include "oshmpi_impl.h"

int shmem_team_my_pe(shmem_team_t team)
{
    OSHMPI_team_t *_team = NULL;
    int rank = 0;

    if (team == SHMEM_TEAM_INVALID) {
        return -1;
    }

    if (team == SHMEM_TEAM_WORLD) {
        return OSHMPI_global.team_world_my_pe;
    } else if (team == SHMEM_TEAM_SHARED) {
        return OSHMPI_global.team_shared_my_pe;
    } else {
        _team = OSHMPI_TEAM_HANDLE_TO_OBJ(team);
        return _team->my_pe;
    }
}

int shmem_team_n_pes(shmem_team_t team)
{
    OSHMPI_team_t *_team = NULL;
    int size = 0;

    if (team == SHMEM_TEAM_INVALID) {
        return -1;
    }

    if (team == SHMEM_TEAM_WORLD) {
        return OSHMPI_global.team_world_n_pes;
    } else if (team == SHMEM_TEAM_SHARED) {
        return OSHMPI_global.team_shared_n_pes;
    } else {
        _team = OSHMPI_TEAM_HANDLE_TO_OBJ(team);
        return _team->n_pes;
    }
}

int shmem_team_get_config(shmem_team_t team, long config_mask, shmem_team_config_t * config)
{
    OSHMPI_team_t *_team = NULL;

    if (team == SHMEM_TEAM_INVALID) {
        return -1;
    }

    if (team == SHMEM_TEAM_WORLD) {
        *config = OSHMPI_global.team_world->config;
    } else if (team == SHMEM_TEAM_SHARED) {
        *config = OSHMPI_global.team_shared->config;
    } else {
        _team = OSHMPI_TEAM_HANDLE_TO_OBJ(team);
        *config = _team->config;
    }

    return SHMEM_SUCCESS;
}

int shmem_team_translate_pe(shmem_team_t src_team, int src_pe, shmem_team_t dest_team)
{
    OSHMPI_team_t *_src_team = NULL;
    OSHMPI_team_t *_dest_team = NULL;
    MPI_Group src_group, dest_group;
    int dest_rank = 0;

    if (src_team == SHMEM_TEAM_INVALID || dest_team == SHMEM_TEAM_INVALID) {
        return -1;
    }

    if (src_team == SHMEM_TEAM_WORLD) {
        src_group = OSHMPI_global.team_world_group;
    } else if (src_team == SHMEM_TEAM_SHARED) {
        src_group = OSHMPI_global.team_shared_group;
    } else {
        _src_team = OSHMPI_TEAM_HANDLE_TO_OBJ(src_team);
        src_group = _src_team->group;
    }
    if (dest_team == SHMEM_TEAM_WORLD) {
        dest_group = OSHMPI_global.team_world_group;
    } else if (dest_team == SHMEM_TEAM_SHARED) {
        dest_group = OSHMPI_global.team_shared_group;
    } else {
        _dest_team = OSHMPI_TEAM_HANDLE_TO_OBJ(dest_team);
        dest_group = _dest_team->group;
    }

    OSHMPI_CALLMPI(MPI_Group_translate_ranks(src_group, 1, &src_pe, dest_group, &dest_rank));
    if (dest_rank == MPI_UNDEFINED) {
        dest_rank = -1;
    }
    return dest_rank;
}

int shmem_team_split_strided(shmem_team_t parent_team, int start, int stride, int size,
                             const shmem_team_config_t * config, long config_mask,
                             shmem_team_t * new_team)
{
    OSHMPI_ASSERT(0);
    return SHMEM_OTHER_ERR;
}

int shmem_team_split_2d(shmem_team_t parent_team, int xrange,
                        const shmem_team_config_t * xaxis_config, long xaxis_mask,
                        shmem_team_t * xaxis_team, const shmem_team_config_t * yaxis_config,
                        long yaxis_mask, shmem_team_t * yaxis_team)
{
    OSHMPI_ASSERT(0);
    return SHMEM_OTHER_ERR;
}

void shmem_team_destroy(shmem_team_t team)
{
    OSHMPI_team_t *_team = NULL;
    if (team == SHMEM_TEAM_INVALID || team == SHMEM_TEAM_WORLD || team == SHMEM_TEAM_SHARED) {
        return;
    }
    _team = OSHMPI_TEAM_HANDLE_TO_OBJ(team);
    OSHMPI_team_destroy(&_team);
}

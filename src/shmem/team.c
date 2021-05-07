/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2021 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#include <shmem.h>
#include "oshmpi_impl.h"

#include <stdbool.h>

int shmem_team_my_pe(shmem_team_t team)
{
    OSHMPI_team_t *_team = NULL;

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
    int rc = SHMEM_SUCCESS;
    OSHMPI_team_t *_parent_team = NULL;
    OSHMPI_team_t *_new_team = NULL;
    bool rank_selected = false;

    if (parent_team == SHMEM_TEAM_INVALID) {
        goto fn_fail;
    }

    if (parent_team == SHMEM_TEAM_WORLD) {
        _parent_team = OSHMPI_global.team_world;
    } else if (parent_team == SHMEM_TEAM_SHARED) {
        _parent_team = OSHMPI_global.team_shared;
    } else {
        _parent_team = OSHMPI_TEAM_HANDLE_TO_OBJ(parent_team);
    }

    /* sanity checks
     * 1. valid start pe rank
     * 2. valid end pe rank */
    if (start < 0 || start >= _parent_team->n_pes) {
        goto fn_fail;
    }

    int end_pe = start + stride * (size - 1);
    if (end_pe < 0 || end_pe >= _parent_team->n_pes) {
        goto fn_fail;
    }

    /* There are two criterias for a PE to be included in the split team
     * 1. rank of the PE in parent team need to be between start and end (calculated using stride
     * and size).
     * 2. rank of the PE must be selected by the (start, stride, size). */
    rank_selected = (_parent_team->my_pe >= start) && (_parent_team->my_pe <= end_pe)
        && ((_parent_team->my_pe - start) % stride == 0);

    OSHMPI_team_split(_parent_team, (rank_selected) ? 1 : MPI_UNDEFINED, &_new_team);

    if (_new_team != NULL) {
        /* handle config if there is config and config_mask != 0 */
        if (config && config_mask != 0) {
            _new_team->config = *config;
        }
    }
    *new_team = OSHMPI_TEAM_OBJ_TO_HANDLE(_new_team);

  fn_exit:
    return rc;
  fn_fail:
    *new_team = SHMEM_TEAM_INVALID;
    rc = SHMEM_OTHER_ERR;
    goto fn_exit;
}

int shmem_team_split_2d(shmem_team_t parent_team, int xrange,
                        const shmem_team_config_t * xaxis_config, long xaxis_mask,
                        shmem_team_t * xaxis_team, const shmem_team_config_t * yaxis_config,
                        long yaxis_mask, shmem_team_t * yaxis_team)
{
    int rc = SHMEM_SUCCESS;
    OSHMPI_team_t *_parent_team = NULL;
    OSHMPI_team_t *_new_x_team = NULL;
    OSHMPI_team_t *_new_y_team = NULL;

    if (parent_team == SHMEM_TEAM_INVALID) {
        goto fn_fail;
    }

    if (parent_team == SHMEM_TEAM_WORLD) {
        _parent_team = OSHMPI_global.team_world;
    } else if (parent_team == SHMEM_TEAM_SHARED) {
        _parent_team = OSHMPI_global.team_shared;
    } else {
        _parent_team = OSHMPI_TEAM_HANDLE_TO_OBJ(parent_team);
    }

    /* X, Y position is used for determine the color of the sub group:
     * 1. PEs in the same X team should have the same Y position (same row)
     * 2. PEs in the same Y team should have the same X position (same col) */
    int x_pos = _parent_team->my_pe % xrange;
    int y_pos = _parent_team->my_pe / xrange;

    OSHMPI_team_split(_parent_team, y_pos, &_new_x_team);
    OSHMPI_team_split(_parent_team, x_pos, &_new_y_team);

    if (_new_x_team != NULL) {
        if (xaxis_config && xaxis_mask != 0) {
            _new_x_team->config = *xaxis_config;
        }
    }
    if (_new_y_team != NULL) {
        if (yaxis_config && yaxis_mask != 0) {
            _new_y_team->config = *yaxis_config;
        }
    }

    *xaxis_team = OSHMPI_TEAM_OBJ_TO_HANDLE(_new_x_team);
    *yaxis_team = OSHMPI_TEAM_OBJ_TO_HANDLE(_new_y_team);

  fn_exit:
    return rc;
  fn_fail:
    *xaxis_team = SHMEM_TEAM_INVALID;
    *yaxis_team = SHMEM_TEAM_INVALID;
    rc = SHMEM_OTHER_ERR;
    goto fn_exit;
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

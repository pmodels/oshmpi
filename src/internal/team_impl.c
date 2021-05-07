/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2021 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include "oshmpi_impl.h"

int OSHMPI_team_create(OSHMPI_team_t ** team)
{
    int rc = SHMEM_SUCCESS;
    OSHMPI_team_t *new_team = NULL;

    new_team = (OSHMPI_team_t *) OSHMPIU_malloc(sizeof(OSHMPI_team_t));

    if (new_team == NULL) {
        rc = SHMEM_OTHER_ERR;
        goto fn_fail;
    }

    new_team->my_pe = -1;
    new_team->n_pes = -1;
    new_team->config.num_contexts = 0;
    new_team->comm = MPI_COMM_NULL;
    new_team->group = MPI_GROUP_NULL;

    *team = new_team;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

void OSHMPI_team_destroy(OSHMPI_team_t ** team)
{
    OSHMPI_CALLMPI(MPI_Group_free(&((*team)->group)));
    OSHMPI_CALLMPI(MPI_Comm_free(&((*team)->comm)));
    OSHMPIU_free(*team);
    *team = NULL;
}

int OSHMPI_team_split(OSHMPI_team_t * team, int color, OSHMPI_team_t ** new_team)
{
    int rc = SHMEM_SUCCESS;
    OSHMPI_team_t *_new_team = NULL;
    MPI_Comm new_comm;
    MPI_Group new_group;

    OSHMPI_CALLMPI(MPI_Comm_split(team->comm, color, team->my_pe, &new_comm));

    if (new_comm != MPI_COMM_NULL) {
        OSHMPI_CALLMPI(MPI_Comm_group(new_comm, &new_group));
        rc = OSHMPI_team_create(&_new_team);
        _new_team->comm = new_comm;
        _new_team->group = new_group;
        OSHMPI_CALLMPI(MPI_Comm_size(_new_team->comm, &_new_team->n_pes));
        OSHMPI_CALLMPI(MPI_Comm_rank(_new_team->comm, &_new_team->my_pe));
        *new_team = (shmem_team_t) _new_team;
    } else {
        *new_team = SHMEM_TEAM_INVALID;
    }
    return rc;
}

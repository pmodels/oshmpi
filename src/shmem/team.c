/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2021 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#include <shmem.h>
#include "oshmpi_impl.h"

int shmem_team_my_pe(shmem_team_t team)
{
    OSHMPI_ASSERT(0);
    return SHMEM_OTHER_ERR;
}

int shmem_team_n_pes(shmem_team_t team)
{
    OSHMPI_ASSERT(0);
    return SHMEM_OTHER_ERR;
}

int shmem_team_get_config(shmem_team_t team, long config_mask, shmem_team_config_t * config)
{
    OSHMPI_ASSERT(0);
    return SHMEM_OTHER_ERR;
}

int shmem_team_translate_pe(shmem_team_t src_team, int src_pe, shmem_team_t dest_team)
{
    OSHMPI_ASSERT(0);
    return SHMEM_OTHER_ERR;
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
    OSHMPI_ASSERT(0);
}

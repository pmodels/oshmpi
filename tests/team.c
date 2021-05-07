/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>
#include <shmemx.h>

#define NSPACES 10

int main(int argc, char *argv[])
{
    int mype, i, total_pes;
    int num_err = 0;

    shmem_init();

    total_pes = shmem_n_pes();
    mype = shmem_my_pe();

    if (total_pes != shmem_team_n_pes(SHMEM_TEAM_WORLD)) {
        fprintf(stdout, "Failed in getting n_pes for TEAM_WORLD\n");
        num_err++;
    }
    if (mype != shmem_team_my_pe(SHMEM_TEAM_WORLD)) {
        fprintf(stdout, "Failed in getting my_pe for TEAM_WORLD\n");
        num_err++;
    }

    shmem_team_t dup_team;

    shmem_team_split_strided(SHMEM_TEAM_WORLD, 0, 1, total_pes, NULL, 0L, &dup_team);
    if (mype != shmem_team_my_pe(dup_team)) {
        fprintf(stdout, "Failed in creating dup team\n");
        num_err++;
    }

    shmem_team_destroy(dup_team);

    shmem_team_t split_team;

    shmem_team_split_strided(SHMEM_TEAM_WORLD, 0, 2, total_pes / 2, NULL, 0L, &split_team);
    if (split_team != SHMEM_TEAM_INVALID) {
        if (mype / 2 != shmem_team_my_pe(split_team)) {
            fprintf(stdout, "Failed in creating odd even split team\n");
            num_err++;
        }
    }

    shmem_team_destroy(split_team);

    shmem_team_t x_team;
    shmem_team_t y_team;

    shmem_team_split_2d(SHMEM_TEAM_WORLD, 2, NULL, 0, &x_team, NULL, 0, &y_team);

    if (2 != shmem_team_n_pes(x_team) && 1 != shmem_team_n_pes(x_team)) {
        printf("%d\n", shmem_team_n_pes(x_team));
        fprintf(stdout, "Failed in creating 2d split team: x team\n");
        num_err++;
    }
    if (((total_pes + 1) / 2 != shmem_team_n_pes(y_team))
        && ((total_pes + 1) / 2 - 1 != shmem_team_n_pes(y_team))) {
        fprintf(stdout, "Failed in creating 2d split team: y team\n");
        num_err++;
    }

    shmem_team_destroy(x_team);
    shmem_team_destroy(y_team);

    shmem_finalize();

    if (mype == 0) {
        if (num_err == 0) {
            fprintf(stdout, "Passed\n");
        }
        fflush(stderr);
    }
    return 0;
}

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
    int mype, i;

    shmem_init();
    mype = shmem_my_pe();

    shmemx_space_config_t space_config;
    shmemx_space_t spaces[NSPACES];

    space_config.sheap_size = 1 << 20;
    space_config.num_contexts = 0;

    for (i = 0; i < NSPACES; i++) {
        shmemx_space_create(space_config, &spaces[i]);
        shmemx_space_attach(spaces[i]);
    }

    int *sbuf = shmemx_space_malloc(spaces[0], 8192);
    int *rbuf = shmemx_space_malloc(spaces[1], 8192);

    shmem_free(sbuf);
    shmem_free(rbuf);

    for (i = 0; i < NSPACES; i++) {
        shmemx_space_detach(spaces[i]);
        shmemx_space_destroy(spaces[i]);
    }

    shmem_finalize();

    if (mype == 0) {
        fprintf(stdout, "Passed\n");
        fflush(stderr);
    }
    return 0;
}

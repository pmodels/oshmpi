/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>

int main(int argc, char *argv[])
{
    int mype, npes;
    int nerrs = 0;

    shmem_init();

    mype = shmem_my_pe();
    npes = shmem_n_pes();

    fprintf(stdout, "Hello world ! I am pe %d in %d\n", mype, npes);
    fflush(stdout);

    if (mype < 0 || npes <= 0 || mype >= npes) {
        fprintf(stderr, "Invalid mype %d or npes %d\n", mype, npes);
        fflush(stderr);
        nerrs++;
    }

    shmem_finalize();
    return 0;
}

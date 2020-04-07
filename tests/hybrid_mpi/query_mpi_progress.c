/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>
#include <shmemx.h>
#include <mpi.h>

int a[65536];
int main(int argc, char *argv[])
{
    int provided;

    /* OSHMPI relies on async thread for progress
     * thus MPI thread multiple safety is required. */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    shmem_init();

    int mype = shmem_my_pe();

    if (!shmemx_query_interoperability(SHMEMX_PROGRESS_MPI))
        shmem_global_exit(EXIT_FAILURE);

    static int b = 0;
    if (mype == 0) {
        MPI_Request req = MPI_REQUEST_NULL;
        MPI_Isend(a, 65536, MPI_INT, 1, 0, MPI_COMM_WORLD, &req);

        while (shmem_int_atomic_fetch(&b, 0) != 1);

        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else if (mype == 1) {
        MPI_Recv(a, 65536, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        shmem_int_atomic_set(&b, 1, 0);
    }

    if (mype == 0) {
        fprintf(stdout, "Passed\n");
        fflush(stderr);
    }

    shmem_finalize();
    MPI_Finalize();

    return 0;
}

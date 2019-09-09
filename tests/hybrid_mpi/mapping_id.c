/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <shmem.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    static long pSync[SHMEM_COLLECT_SYNC_SIZE];
    for (int i = 0; i < SHMEM_COLLECT_SYNC_SIZE; i++)
        pSync[i] = SHMEM_SYNC_VALUE;

    int mpi_provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_provided);
    shmem_init();

    int mype = shmem_my_pe();
    int npes = shmem_n_pes();

    static int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int *mpi_ranks = shmem_calloc(npes, sizeof(int));

    shmem_collect32(mpi_ranks, &myrank, 1, 0, 0, npes, pSync);

    if (mype == 0)
        for (int i = 0; i < npes; i++)
            printf("PE %d's MPI rank is %d\n", i, mpi_ranks[i]);

    shmem_free(mpi_ranks);

    if (mype == 0) {
        fprintf(stdout, "Passed\n");
        fflush(stderr);
    }

    shmem_finalize();
    MPI_Finalize();

    return 0;
}

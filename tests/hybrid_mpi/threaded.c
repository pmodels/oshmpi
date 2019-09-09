/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int mype, rank;
    int shmem_provided, mpi_provided;
    int shmem_ret = 0, mpi_ret = MPI_SUCCESS;

    mpi_ret = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_provided);
    shmem_ret = shmem_init_thread(SHMEM_THREAD_MULTIPLE, &shmem_provided);

    mype = shmem_my_pe();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (shmem_ret || shmem_provided != SHMEM_THREAD_MULTIPLE) {
        printf("shmem_init_thread returned %d on pe %d, expected THREAD_MULTIPLE(%d),"
               "but provided %d\n", shmem_ret, mype, SHMEM_THREAD_MULTIPLE, shmem_provided);
    } else if (mpi_ret != MPI_SUCCESS || mpi_provided != MPI_THREAD_MULTIPLE) {
        printf("MPI_Init_thread returned %d on rank %d, expected THREAD_MULTIPLE(%d),"
               "but provided %d\n", shmem_ret, rank, MPI_THREAD_MULTIPLE, mpi_provided);
    } else {
        printf("Both shmem_init_thread and MPI_Init_thread support "
               "THREAD_MULTIPLE on pe %d\n", mype);
    }

    if (mype == 0) {
        fprintf(stdout, "Passed\n");
        fflush(stderr);
    }

    shmem_finalize();
    MPI_Finalize();

    return 0;
}

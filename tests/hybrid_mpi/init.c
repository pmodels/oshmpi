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
#ifdef TEST_MPI_INIT_FIRST
    int provided;
    /* OSHMPI relies on async thread for progress
     * thus MPI thread multiple safety is required. */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    fprintf(stdout, "1. MPI_Init called\n");

    shmem_init();
    fprintf(stdout, "2. shmem_init called\n");
#else
    shmem_init();
    fprintf(stdout, "1. shmem_init called\n");

    /* MPI may already be initialized in shmem_init. */
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized)
        MPI_Init(&argc, &argv);

    fprintf(stdout, "2. MPI_Init %s\n", mpi_initialized ? "skipped" : "called");
#endif

#ifdef TEST_MPI_FINALIZE_FIRST
    /* OSHMPI does not support this order. */
    MPI_Finalize();
    fprintf(stdout, "3. MPI_Finalize called\n");

    shmem_finalize();
    fprintf(stdout, "4. shmem_finalize called\n");
#else
    shmem_finalize();
    fprintf(stdout, "3. shmem_finalize called\n");

    /* MPI may already be finalized in shmem_finalize. */
    int mpi_finalized = 0;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized)
        MPI_Finalize();
    fprintf(stdout, "4. MPI_Finalize %s\n", mpi_finalized ? "skipped" : "called");
#endif

    fprintf(stdout, "Passed\n");
    fflush(stdout);

    return 0;
}

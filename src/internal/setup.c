/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include "oshmpi_impl.h"

OSHMPI_global_t OSHMPI_global = { 0 };

int OSHMPI_initialize_thread(int required, int *provided)
{
    int mpi_errno = MPI_SUCCESS;
    int mpi_provided = 0;

    if (required < SHMEM_THREAD_SINGLE || required > SHMEM_THREAD_MULTIPLE)
        OSHMPI_ERR_ABORT("Unknown OpenSHMEM thread support level: %d\n", required);

    MPI_Init_thread(NULL, NULL, required, &mpi_provided);
    if (mpi_provided != required) {
        OSHMPI_ERR_ABORT("The MPI library does not support the required thread support:"
                         "required: %s, provided: %s.\n",
                         OSHMPI_thread_level_str(required), OSHMPI_thread_level_str(mpi_provided));
    }

    MPI_Comm_size(MPI_COMM_WORLD, &OSHMPI_global.world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &OSHMPI_global.world_rank);

    OSHMPI_global.is_initialized = 1;

    if (provided)
        *provided = mpi_provided;

    return mpi_errno;
}

int OSHMPI_finalize(void)
{
    int mpi_errno = MPI_SUCCESS;

    MPI_Finalize();

    return mpi_errno;
}

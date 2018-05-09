/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include "oshmpi_impl.h"

void shmem_init(void)
{
    OSHMPI_initialize_thread(SHMEM_THREAD_SINGLE, NULL);
}

int shmem_my_pe(void)
{
    return OSHMPI_global.world_rank;
}

int shmem_n_pes(void)
{
    return OSHMPI_global.world_size;
}

void shmem_finalize(void)
{
    OSHMPI_finalize();
}

void shmem_global_exit(int status)
{

}

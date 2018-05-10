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
    MPI_Abort(OSHMPI_global.comm_world, status);
}

void shmem_init_thread(int requested, int *provided)
{
    OSHMPI_initialize_thread(requested, provided);
}

void shmem_query_thread(int *provided)
{
    *provided = OSHMPI_global.thread_level;
}

int shmem_pe_accessible(int pe)
{
    return (pe >= 0 && pe < OSHMPI_global.world_size) ? 1 : 0;
}

int shmem_addr_accessible(const void *addr, int pe)
{
    return (OSHMPI_global.symm_heap_base <= addr &&
            (MPI_Aint) addr < (MPI_Aint) OSHMPI_global.symm_heap_base +
            OSHMPI_global.symm_heap_size && (pe >= 0 && pe < OSHMPI_global.world_size)) ? 1 : 0;
}

void *shmem_ptr(const void *dest, int pe)
{
    /* Do not support load/store for other processes */
    return (pe == OSHMPI_global.world_rank) ? (void *) dest : NULL;
}

void shmem_info_get_version(int *major, int *minor)
{
    *major = SHMEM_MAJOR_VERSION;
    *minor = SHMEM_MINOR_VERSION;
}

void shmem_info_get_name(char *name)
{
    strncpy(name, SHMEM_VENDOR_STRING, SHMEM_MAX_NAME_LEN);
    name[SHMEM_MAX_NAME_LEN - 1] = '\0';        /* Ensure string is null terminated */
}


/* (deprecated APIs) */

void start_pes(int npes)
{
    /* Skip if already initialized */
    if (OSHMPI_global.is_initialized)
        return;

    OSHMPI_initialize_thread(SHMEM_THREAD_SINGLE, NULL);

    /* Mark as initialized by start_pes */
    OSHMPI_global.is_start_pes_initialized = 1;

    /* Register implicit finalization for programs started by start_pes. */
    atexit(OSHMPI_implicit_finalize);
}

int _my_pe(void)
{
    return shmem_my_pe();
}

int _num_pes(void)
{
    return shmem_n_pes();
}

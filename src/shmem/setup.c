/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include "oshmpi_impl.h"

void shmem_init(void)
{
    OSHMPI_NOINLINE_RECURSIVE()
        OSHMPI_initialize_thread(OSHMPI_DEFAULT_THREAD_SAFETY, NULL);
    if (OSHMPI_env.version && OSHMPI_global.team_world_my_pe == 0)
        OSHMPI_PRINTF("SHMEM library version:\n"
                      "    SHMEM_MAJOR_VERSION  %d\n"
                      "    SHMEM_MINOR_VERSION  %d\n"
                      "    OSHMPI_VERSION       %s\n\n",
                      SHMEM_MAJOR_VERSION, SHMEM_MINOR_VERSION, OSHMPI_VERSION);
}

int shmem_my_pe(void)
{
    return OSHMPI_global.team_world_my_pe;
}

int shmem_n_pes(void)
{
    return OSHMPI_global.team_world_n_pes;
}

void shmem_finalize(void)
{
    OSHMPI_NOINLINE_RECURSIVE()
        OSHMPI_finalize();
}

void shmem_global_exit(int status)
{
    OSHMPI_global_exit(status);
    while (1);  /* Mark non-stop to avoid a c11 warning about noreturn. */
}

int shmem_init_thread(int requested, int *provided)
{
    OSHMPI_NOINLINE_RECURSIVE()
        OSHMPI_initialize_thread(requested, provided);

    if (OSHMPI_env.version && OSHMPI_global.team_world_my_pe == 0)
        OSHMPI_PRINTF("SHMEM library version:\n"
                      "    SHMEM_MAJOR_VERSION  %d\n"
                      "    SHMEM_MINOR_VERSION  %d\n"
                      "    OSHMPI_VERSION       %s\n\n",
                      SHMEM_MAJOR_VERSION, SHMEM_MINOR_VERSION, OSHMPI_VERSION);

    /* Returns 0 upon success; otherwise, it returns a non-zero value.
     * For any MPI internal error we expect the default behavior is abort from MPI.
     * Thus we check only the provided thread level here. */
    return (requested <= *provided) ? SHMEM_SUCCESS : SHMEM_OTHER_ERR;
}

void shmem_query_thread(int *provided)
{
    *provided = OSHMPI_global.thread_level;
}

int shmem_pe_accessible(int pe)
{
    return (pe >= 0 && pe < OSHMPI_global.team_world_n_pes) ? 1 : 0;
}

int shmem_addr_accessible(const void *addr, int pe)
{
    return ((OSHMPI_sobj_check_range(addr, OSHMPI_global.symm_heap_attr) ||
             OSHMPI_sobj_check_range(addr, OSHMPI_global.symm_data_attr)) &&
            (pe >= 0 && pe < OSHMPI_global.team_world_n_pes)) ? 1 : 0;
}

void *shmem_ptr(const void *dest, int pe)
{
    /* Do not support load/store for other processes */
    return (pe == OSHMPI_global.team_world_my_pe) ? (void *) dest : NULL;
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

    if (OSHMPI_env.version && OSHMPI_global.team_world_my_pe == 0)
        OSHMPI_PRINTF("SHMEM library version:\n"
                      "    SHMEM_MAJOR_VERSION  %d\n"
                      "    SHMEM_MINOR_VERSION  %d\n"
                      "    OSHMPI_VERSION       %s\n\n",
                      SHMEM_MAJOR_VERSION, SHMEM_MINOR_VERSION, OSHMPI_VERSION);
}

int _my_pe(void)
{
    return shmem_my_pe();
}

int _num_pes(void)
{
    return shmem_n_pes();
}

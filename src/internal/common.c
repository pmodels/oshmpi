/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include <shmemx.h>
#include "oshmpi_impl.h"

void OSHMPI_sobj_init_attr(OSHMPI_sobj_attr_t * sobj_attr, shmemx_memkind_t memkind,
                           void *base, MPI_Aint size)
{
    memset(sobj_attr, 0, sizeof(OSHMPI_sobj_attr_t));
    sobj_attr->base = base;
    sobj_attr->size = size;
    sobj_attr->memkind = memkind;

    /* Remaining attributes:
     * - handle is set by each caller
     * - symm_flag and offsets are set at later collective call by each caller */
}

void OSHMPI_sobj_symm_info_allgather(OSHMPI_sobj_attr_t * sobj_attr, int *symm_flag)
{
#if defined(OSHMPI_ENABLE_DYNAMIC_WIN)
    OSHMPIU_check_symm_mem(sobj_attr->base, symm_flag, &sobj_attr->base_offsets);
#endif
}

void OSHMPI_sobj_symm_info_dbgprint(OSHMPI_sobj_attr_t * sobj_attr)
{
    OSHMPI_DBGMSG("base %p, size %ld, handle 0x%x (symmbit %d)\n", sobj_attr->base, sobj_attr->size,
                  sobj_attr->handle, OSHMPI_SOBJ_HANDLE_GET_SYMMBIT(sobj_attr->handle));
#if defined(OSHMPI_ENABLE_DYNAMIC_WIN)
    int i;
    if (!OSHMPI_SOBJ_HANDLE_CHECK_SYMMBIT(sobj_attr->handle))
        for (i = 0; i < OSHMPI_global.world_size; i++)
            OSHMPI_DBGMSG("    offset[%d]=0x%lx\n", i, sobj_attr->base_offsets[i]);
#endif
}

void OSHMPI_sobj_destroy_attr(OSHMPI_sobj_attr_t * sobj_attr)
{
#if defined(OSHMPI_ENABLE_DYNAMIC_WIN)
    if (!OSHMPI_SOBJ_HANDLE_CHECK_SYMMBIT(sobj_attr->handle))
        OSHMPIU_free(sobj_attr->base_offsets);
#endif
}

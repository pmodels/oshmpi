/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#include <shmem.h>
#include "oshmpi_impl.h"

void shmem_barrier_all(void)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_barrier(int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_sync_all(void)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_sync(int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_broadcast32(void *dest, const void *source, size_t nelems, int PE_root, int PE_start,
                       int logPE_stride, int PE_size, long *pSync)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_broadcast64(void *dest, const void *source, size_t nelems, int PE_root, int PE_start,
                       int logPE_stride, int PE_size, long *pSync)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_collect32(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                     int PE_size, long *pSync)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_collect64(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                     int PE_size, long *pSync)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_fcollect32(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                      int PE_size, long *pSync)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_fcollect64(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                      int PE_size, long *pSync)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_alltoall32(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                      int PE_size, long *pSync)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_alltoall64(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                      int PE_size, long *pSync)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_alltoalls32(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems,
                       int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_alltoalls64(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems,
                       int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2022 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#include <shmem.h>
#include "oshmpi_impl.h"

void shmem_barrier(int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    OSHMPI_barrier(PE_start, logPE_stride, PE_size);
}

void shmem_sync_aset(int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    /* Deprecated API */
    OSHMPI_sync(PE_start, logPE_stride, PE_size);
}

void shmem_broadcast32(void *dest, const void *source, size_t nelems, int PE_root, int PE_start,
                       int logPE_stride, int PE_size, long *pSync)
{
    OSHMPI_broadcast(dest, source, nelems, OSHMPI_MPI_COLL32_T, PE_root, PE_start, logPE_stride,
                     PE_size);
}

void shmem_broadcast64(void *dest, const void *source, size_t nelems, int PE_root, int PE_start,
                       int logPE_stride, int PE_size, long *pSync)
{
    OSHMPI_broadcast(dest, source, nelems, OSHMPI_MPI_COLL64_T, PE_root, PE_start, logPE_stride,
                     PE_size);
}

void shmem_collect32(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                     int PE_size, long *pSync)
{
    OSHMPI_collect(dest, source, nelems, OSHMPI_MPI_COLL32_T, PE_start, logPE_stride, PE_size);
}

void shmem_collect64(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                     int PE_size, long *pSync)
{
    OSHMPI_collect(dest, source, nelems, OSHMPI_MPI_COLL64_T, PE_start, logPE_stride, PE_size);
}

void shmem_fcollect32(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                      int PE_size, long *pSync)
{
    OSHMPI_fcollect(dest, source, nelems, OSHMPI_MPI_COLL32_T, PE_start, logPE_stride, PE_size);
}

void shmem_fcollect64(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                      int PE_size, long *pSync)
{
    OSHMPI_fcollect(dest, source, nelems, OSHMPI_MPI_COLL64_T, PE_start, logPE_stride, PE_size);
}

void shmem_alltoall32(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                      int PE_size, long *pSync)
{
    OSHMPI_alltoall(dest, source, nelems, OSHMPI_MPI_COLL32_T, PE_start, logPE_stride, PE_size);
}

void shmem_alltoall64(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                      int PE_size, long *pSync)
{
    OSHMPI_alltoall(dest, source, nelems, OSHMPI_MPI_COLL64_T, PE_start, logPE_stride, PE_size);
}

void shmem_alltoalls32(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems,
                       int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    OSHMPI_alltoalls(dest, source, dst, sst, nelems, OSHMPI_MPI_COLL32_T, PE_start, logPE_stride,
                     PE_size);
}

void shmem_alltoalls64(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems,
                       int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    OSHMPI_alltoalls(dest, source, dst, sst, nelems, OSHMPI_MPI_COLL64_T, PE_start, logPE_stride,
                     PE_size);
}

/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#ifndef OSHMPI_SHMEM_H
#define OSHMPI_SHMEM_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <mpi.h>

/* OSHMPI_VERSION is the version string. OSHMPI_NUMVERSION is the
 * numeric version that can be used in numeric comparisons.
 *
 * OSHMPI_VERSION uses the following format:
 * Version: [MAJ].[MIN].[REV][EXT][EXT_NUMBER]
 * Example: 1.0.7rc1 has
 *          MAJ = 1
 *          MIN = 0
 *          REV = 7
 *          EXT = rc
 *          EXT_NUMBER = 1
 *
 * OSHMPI_NUMVERSION will convert EXT to a format number:
 *          ALPHA (a) = 0
 *          BETA (b)  = 1
 *          RC (rc)   = 2
 *          PATCH (p) = 3
 * Regular releases are treated as patch 0
 *
 * Numeric version will have 1 digit for MAJ, 2 digits for MIN, 2
 * digits for REV, 1 digit for EXT and 2 digits for EXT_NUMBER. So,
 * 1.0.7rc1 will have the numeric version 10007201.
 */
#define OSHMPI_VERSION "@OSHMPI_VERSION@"
#define OSHMPI_NUMVERSION @OSHMPI_NUMVERSION@
#define OSHMPI_RELEASE_DATE "@OSHMPI_RELEASE_DATE@"
#define OSHMPI_BUILD_INFO "@CONFIGURE_ARGS_CLEAN@"

#define  SHMEM_MAJOR_VERSION 1
#define _SHMEM_MAJOR_VERSION SHMEM_MAJOR_VERSION
#define  SHMEM_MINOR_VERSION 4
#define _SHMEM_MINOR_VERSION SHMEM_MINOR_VERSION
#define  SHMEM_MAX_NAME_LEN  256
#define _SHMEM_MAX_NAME_LEN  SHMEM_MAX_NAME_LEN
#define  SHMEM_VENDOR_STRING "OSHMPI"
#define _SHMEM_VENDOR_STRING SHMEM_VENDOR_STRING

/* Return code of SHMEM routine */
#define SHMEM_SUCCESS 0
#define SHMEM_NO_CTX -1
#define SHMEM_OTHER_ERR -2

#define SHMEM_THREAD_SINGLE MPI_THREAD_SINGLE
#define SHMEM_THREAD_FUNNELED MPI_THREAD_FUNNELED
#define SHMEM_THREAD_SERIALIZED MPI_THREAD_SERIALIZED
#define SHMEM_THREAD_MULTIPLE MPI_THREAD_MULTIPLE

/* Context option constants (long) and type */
#define SHMEM_CTX_SERIALIZED 0x001001L
#define SHMEM_CTX_PRIVATE 0x001002L
#define SHMEM_CTX_NOSTORE 0x001003L

#define SHMEM_CTX_DEFAULT -1
typedef int shmem_ctx_t;

/* Collective constants */
#define SHMEM_SYNC_VALUE 0L
#define SHMEM_SYNC_SIZE 1
#define SHMEM_BARRIER_SYNC_SIZE 1
#define SHMEM_BCAST_SYNC_SIZE 1
#define SHMEM_REDUCE_SYNC_SIZE 1
#define SHMEM_COLLECT_SYNC_SIZE 1
#define SHMEM_ALLTOALL_SYNC_SIZE 1
#define SHMEM_ALLTOALLS_SYNC_SIZE 1
#define SHMEM_REDUCE_MIN_WRKDATA_SIZE 1
/* (deprecated constants) */
#define _SHMEM_SYNC_VALUE SHMEM_SYNC_VALUE
#define _SHMEM_BARRIER_SYNC_SIZE SHMEM_BARRIER_SYNC_SIZE
#define _SHMEM_BCAST_SYNC_SIZE SHMEM_BCAST_SYNC_SIZE
#define _SHMEM_REDUCE_SYNC_SIZE SHMEM_REDUCE_SYNC_SIZE
#define _SHMEM_COLLECT_SYNC_SIZE SHMEM_COLLECT_SYNC_SIZE
#define _SHMEM_REDUCE_MIN_WRKDATA_SIZE SHMEM_REDUCE_MIN_WRKDATA_SIZE

/* Point-to-Point comparison constants */
#define SHMEM_CMP_EQ 0x100001
#define SHMEM_CMP_NE 0x100002
#define SHMEM_CMP_GT 0x100003
#define SHMEM_CMP_GE 0x100004
#define SHMEM_CMP_LT 0x100005
#define SHMEM_CMP_LE 0x100006
/* (deprecated constants) */
#define _SHMEM_CMP_EQ SHMEM_CMP_EQ
#define _SHMEM_CMP_NE SHMEM_CMP_NE
#define _SHMEM_CMP_GT SHMEM_CMP_GT
#define _SHMEM_CMP_GE SHMEM_CMP_GE
#define _SHMEM_CMP_LT SHMEM_CMP_LT
#define _SHMEM_CMP_LE SHMEM_CMP_LE

/* -- Library Setup, Exit, and Query Routines -- */
void shmem_init(void);
int shmem_my_pe(void);
int shmem_n_pes(void);
void shmem_finalize(void);
void shmem_global_exit(int status);
int shmem_pe_accessible(int pe);
int shmem_addr_accessible(const void *addr, int pe);
void *shmem_ptr(const void *dest, int pe);
void shmem_info_get_version(int *major, int *minor);
void shmem_info_get_name(char *name);
/* (deprecated APIs) */
void start_pes(int npes);
int _my_pe(void);
int _num_pes(void);

/* -- Thread Support -- */
void shmem_init_thread(int requested, int *provided);
void shmem_query_thread(int *provided);

/* -- Memory Management -- */
void *shmem_malloc(size_t size);
void shmem_free(void *ptr);
void *shmem_realloc(void *ptr, size_t size);
void *shmem_align(size_t alignment, size_t size);
void *shmem_calloc(size_t count, size_t size);
/* (deprecated APIs) */
void *shmalloc(size_t size);
void shfree(void *ptr);
void *shrealloc(void *ptr, size_t size);
void *shmemalign(size_t alignment, size_t size);

/* -- Communication Management -- */
int shmem_ctx_create(long options, shmem_ctx_t * ctx);
void shmem_ctx_destroy(shmem_ctx_t ctx);

/* -- RMA and Atomics -- */
void shmem_ctx_putmem(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe);
void shmem_putmem(void *dest, const void *source, size_t nelems, int pe);
void shmem_ctx_getmem(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe);
void shmem_getmem(void *dest, const void *source, size_t nelems, int pe);
void shmem_ctx_putmem_nbi(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe);
void shmem_putmem_nbi(void *dest, const void *source, size_t nelems, int pe);
void shmem_ctx_getmem_nbi(shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe);
void shmem_getmem_nbi(void *dest, const void *source, size_t nelems, int pe);

/* SHMEM_RMA_TYPED_H start */
/* SHMEM_RMA_TYPED_H end */

/* SHMEM_RMA_SIZED_H start */
/* SHMEM_RMA_SIZED_H end */

/* SHMEM_AMO_STD_TYPED_H start */
/* SHMEM_AMO_STD_TYPED_H end */

/* SHMEM_AMO_EXT_TYPED_H start */
/* SHMEM_AMO_EXT_TYPED_H end */

/* SHMEM_AMO_BITWS_TYPED_H start */
/* SHMEM_AMO_BITWS_TYPED_H end */

/* -- Collectives -- */
void shmem_barrier_all(void);
void shmem_barrier(int PE_start, int logPE_stride, int PE_size, long *pSync);
void shmem_sync_all(void);
void shmem_sync(int PE_start, int logPE_stride, int PE_size, long *pSync);
void shmem_broadcast32(void *dest, const void *source, size_t nelems, int PE_root, int PE_start,
                       int logPE_stride, int PE_size, long *pSync);
void shmem_broadcast64(void *dest, const void *source, size_t nelems, int PE_root, int PE_start,
                       int logPE_stride, int PE_size, long *pSync);
void shmem_collect32(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                     int PE_size, long *pSync);
void shmem_collect64(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                     int PE_size, long *pSync);
void shmem_fcollect32(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                      int PE_size, long *pSync);
void shmem_fcollect64(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                      int PE_size, long *pSync);
void shmem_alltoall32(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                      int PE_size, long *pSync);
void shmem_alltoall64(void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,
                      int PE_size, long *pSync);
void shmem_alltoalls32(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems,
                       int PE_start, int logPE_stride, int PE_size, long *pSync);
void shmem_alltoalls64(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems,
                       int PE_start, int logPE_stride, int PE_size, long *pSync);

/* SHMEM_REDUCE_MINMAX_TYPED_H start */
/* SHMEM_REDUCE_MINMAX_TYPED_H end */

/* SHMEM_REDUCE_SUMPROD_TYPED_H start */
/* SHMEM_REDUCE_SUMPROD_TYPED_H end */

/* SHMEM_REDUCE_BITWS_TYPED_H start */
/* SHMEM_REDUCE_BITWS_TYPED_H end */

/* -- Point-To-Point Synchronization -- */
/* SHMEM_P2P_TYPED_H start */
/* SHMEM_P2P_TYPED_H end */

/* (deprecated APIs) */
void shmem_wait_until(long *ivar, int cmp, long cmp_value);
void shmem_wait(long *ivar, long cmp_value);
void shmem_short_wait(short *ivar, short cmp_value);
void shmem_int_wait(int *ivar, int cmp_value);
void shmem_long_wait(long *ivar, long cmp_value);
void shmem_longlong_wait(long long *ivar, long long cmp_value);

/* -- Memory Ordering -- */
void shmem_fence(void);
void shmem_ctx_fence(shmem_ctx_t ctx);
void shmem_quiet(void);
void shmem_ctx_quiet(shmem_ctx_t ctx);

/* -- Distributed Locking -- */
void shmem_clear_lock(long *lock);
void shmem_set_lock(long *lock);
int shmem_test_lock(long *lock);

/* -- Cache Management -- */
/* (deprecated APIs) */
void shmem_clear_cache_inv(void);
void shmem_set_cache_inv(void);
void shmem_clear_cache_line_inv(void *dest);
void shmem_set_cache_line_inv(void *dest);
void shmem_udcflush(void);
void shmem_udcflush_line(void *dest);

#endif /* OSHMPI_SHMEM_H */

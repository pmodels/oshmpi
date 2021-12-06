/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef OSHMPI_UTIL_H
#define OSHMPI_UTIL_H

#include <oshmpiconf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>
#ifdef OSHMPI_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

/* ======================================================================
 * Generic Utility MACROs and inline functions.
 * ====================================================================== */

#ifndef OSHMPI_UNLIKELY
#ifdef HAVE_BUILTIN_EXPECT
#define OSHMPI_UNLIKELY(x_) __builtin_expect(!!(x_),0)
#else
#define OSHMPI_UNLIKELY(x_) (x_)
#endif
#endif /* OSHMPI_UNLIKELY */

#ifndef OSHMPI_LIKELY
#ifdef HAVE_BUILTIN_EXPECT
#define OSHMPI_LIKELY(x_)   __builtin_expect(!!(x_),1)
#else
#define OSHMPI_LIKELY(x_)   (x_)
#endif
#endif /* OSHMPI_LIKELY */

#ifndef OSHMPI_ATTRIBUTE
#ifdef HAVE_GCC_ATTRIBUTE
#define OSHMPI_ATTRIBUTE(a_) __attribute__(a_)
#else
#define OSHMPI_ATTRIBUTE(a_)
#endif
#endif /* OSHMPI_ATTRIBUTE */

#ifndef OSHMPI_ALIGN
#define OSHMPI_ALIGN(val, align) (((unsigned long) (val) + (align) - 1) & ~((align) - 1))
#endif

#ifndef OSHMPI_MAX
#define OSHMPI_MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef OSHMPI_MAX
#define OSHMPI_MAX(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef OSHMPI_STATIC_INLINE_PREFIX
#ifdef OSHMPI_ENABLE_DBG        /* Disable force inline in debug mode */
#define OSHMPI_STATIC_INLINE_PREFIX static inline
#else
#define OSHMPI_STATIC_INLINE_PREFIX OSHMPI_ATTRIBUTE((always_inline)) static inline
#endif
#endif /* OSHMPI_STATIC_INLINE_PREFIX */

#ifndef OSHMPI_DISABLE_ERROR_CHECKING
#define OSHMPI_ASSERT(EXPR) do { if (OSHMPI_UNLIKELY(!(EXPR))){           \
            fprintf(stderr, "OSHMPI assert fail in [%s:%d]: \"%s\"\n",    \
                          __FILE__, __LINE__, #EXPR);               \
            fflush(stderr);                                         \
            MPI_Abort(MPI_COMM_WORLD, -1);                          \
        }} while (0)
#else
#define OSHMPI_ASSERT(EXPR) do {} while (0);
#endif

/* TODO: define consistent error handling & report */
#define OSHMPI_ERR_ABORT(MSG,...) do {                                  \
            fprintf(stderr, "OSHMPI abort in [%s:%d]:"MSG,              \
                          __FILE__, __LINE__, ## __VA_ARGS__);     \
            fflush(stderr);                                        \
            MPI_Abort(MPI_COMM_WORLD, -1);                         \
        } while (0)

#ifndef OSHMPI_DISABLE_DEBUG
#define OSHMPI_DBGMSG(MSG,...) do {                                                   \
            if (OSHMPI_env.debug) {                                                   \
                fprintf(stdout, "OSHMPIDBG[%d] %s: "MSG,                              \
                        OSHMPI_global.team_world_my_pe, __FUNCTION__, ## __VA_ARGS__);\
                fflush(stdout);                                                       \
            }                                                                         \
        } while (0)
#else
#define OSHMPI_DBGMSG(MSG,...) do { } while (0)
#endif

#define OSHMPI_PRINTF(MSG,...) do {                                             \
                fprintf(stdout, MSG, ## __VA_ARGS__); fflush(stdout);           \
        } while (0)

/*  MPI call wrapper.
 *  No consistent error handling is defined in OpenSHMEM. For now,
 *  we simply assume processes abort inside MPI when an MPI error occurs
 *  (MPI default error handler: MPI_ERRORS_ARE_FATAL). */
#define OSHMPI_CALLMPI(fnc_stmt) do {      \
            fnc_stmt;                      \
        } while (0)

#define OSHMPI_CALLMPI_RET(ret, fnc_stmt) do {      \
            ret = fnc_stmt;                         \
        } while (0)

#define OSHMPI_CALLPTHREAD(fnc_stmt) do {  \
            int err = 0;                   \
            err = fnc_stmt;                \
            OSHMPI_ASSERT(err == 0);       \
        } while (0);

#ifdef OSHMPI_ENABLE_CUDA
/*  CUDA call wrapper.
 *  No consistent error handling is defined in OpenSHMEM.
 *  For now, we simply abort. */
#define OSHMPI_CALLCUDA(fnc_stmt) do {            \
            cudaError_t err = cudaSuccess;        \
            err = fnc_stmt;                       \
            if (err != cudaSuccess) OSHMPI_ERR_ABORT("cuda error:%d %s\n", err, cudaGetErrorString(err));    \
        } while (0)
#endif

OSHMPI_STATIC_INLINE_PREFIX void *OSHMPIU_malloc(size_t size)
{
    return malloc(size);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPIU_free(void *buf)
{
    free(buf);
}

OSHMPI_STATIC_INLINE_PREFIX uint64_t OSHMPIU_str_to_size(char *s)
{
    double val;
    char *e;

    val = (double) strtof(s, &e);
    if (e == NULL || *e == '\0')
        return val;

    if (*e == 'K' || *e == 'k')
        val *= 1024.;
    else if (*e == 'M' || *e == 'm')
        val *= 1024. * 1024.;
    else if (*e == 'G' || *e == 'g')
        val *= 1024. * 1024. * 1024.;
    else if (*e == 'T' || *e == 't')
        val *= 1024. * 1024. * 1024. * 1024.;

    /* ceil is required to match the specification example
     * "3.1M is equivalent to the integer value 3250586" */
    uint64_t ret_val = ceil(val);
    return ret_val;
}

/* ======================================================================
 * Convenient helper functions
 * ====================================================================== */

OSHMPI_STATIC_INLINE_PREFIX const char *OSHMPI_thread_level_str(int level)
{
    const char *str = "";
    switch (level) {
        case MPI_THREAD_SINGLE:
            str = "THREAD_SINGLE";
            break;
        case MPI_THREAD_FUNNELED:
            str = "THREAD_FUNNELED";
            break;
        case MPI_THREAD_SERIALIZED:
            str = "THREAD_SERIALIZED";
            break;
        case MPI_THREAD_MULTIPLE:
            str = "THREAD_MULTIPLE";
            break;
    }
    return str;
}

void OSHMPIU_initialize_symm_mem(MPI_Comm comm_world);
int OSHMPIU_allocate_symm_mem(MPI_Aint size, void **local_addr_ptr);
void OSHMPIU_free_symm_mem(void *local_addr, MPI_Aint size);
void OSHMPIU_check_symm_mem(void *local_addr, int *symm_flag_ptr, MPI_Aint ** all_addrs_ptr);

typedef struct OSUMPIU_mempool {
    void *base;
    size_t size;
    size_t chunk_size;
    int nchunks;
    unsigned int *chunks_nused; /* array of counters indicating the number of contiguous chunks
                                 * being used. The counter is set only at the first chunk of every
                                 * used region which contains one or multiple contiguous chunks. */
} OSUMPIU_mempool_t;

/* Memory pool routines (thread unsafe) */
void OSHMPIU_mempool_init(OSUMPIU_mempool_t * mem_pool, void *base, size_t aligned_mem_size,
                          size_t chunk_size);
void OSHMPIU_mempool_destroy(OSUMPIU_mempool_t * mem_pool);
void *OSHMPIU_mempool_alloc(OSUMPIU_mempool_t * mem_pool, size_t size);
void OSHMPIU_mempool_free(OSUMPIU_mempool_t * mem_pool, void *ptr);

/* ======================================================================
 * Atomic helper functions
 * ====================================================================== */

OSHMPI_STATIC_INLINE_PREFIX int OSHMPIU_single_thread_cas_int(int *val, int old, int new)
{
    int prev;
    prev = *val;
    if (prev == old)
        *val = new;
    return prev;
}

OSHMPI_STATIC_INLINE_PREFIX int OSHMPIU_single_thread_finc_int(int *val)
{
    int prev;
    prev = *val;
    (*val)++;
    return prev;
}

#if defined(OSHMPI_ENABLE_THREAD_MULTIPLE)
#include "opa_primitives.h"
typedef OPA_int_t OSHMPIU_atomic_flag_t;
#define OSHMPIU_ATOMIC_FLAG_STORE(flag, val) OPA_store_int(&(flag), val)
#define OSHMPIU_ATOMIC_FLAG_LOAD(flag) OPA_load_int(&(flag))
#define OSHMPIU_ATOMIC_FLAG_CAS(flag, old, new) OPA_cas_int(&(flag), (old), (new))

typedef OPA_int_t OSHMPIU_atomic_cnt_t;
#define OSHMPIU_ATOMIC_CNT_STORE(cnt, val) OPA_store_int(&(cnt), val)
#define OSHMPIU_ATOMIC_CNT_LOAD(cnt) OPA_load_int(&(cnt))
#define OSHMPIU_ATOMIC_CNT_INCR(cnt) OPA_incr_int(&(cnt))
#define OSHMPIU_ATOMIC_CNT_DECR(cnt) OPA_decr_int(&(cnt))
#define OSHMPIU_ATOMIC_CNT_FINC(cnt) OPA_fetch_and_incr_int(&(cnt))
#else
typedef unsigned int OSHMPIU_atomic_flag_t;
#define OSHMPIU_ATOMIC_FLAG_STORE(flag, val) do {(flag) = (val);} while (0)
#define OSHMPIU_ATOMIC_FLAG_LOAD(flag) (flag)
#define OSHMPIU_ATOMIC_FLAG_CAS(flag, old, new) OSHMPIU_single_thread_cas_int(&(flag), old, new)

typedef unsigned int OSHMPIU_atomic_cnt_t;
#define OSHMPIU_ATOMIC_CNT_STORE(cnt, val) do {(cnt) = (val);} while (0)
#define OSHMPIU_ATOMIC_CNT_LOAD(cnt) (cnt)
#define OSHMPIU_ATOMIC_CNT_INCR(cnt) do {(cnt)++;} while (0)
#define OSHMPIU_ATOMIC_CNT_DECR(cnt) do {(cnt)--;} while (0)
#define OSHMPIU_ATOMIC_CNT_FINC(cnt) OSHMPIU_single_thread_finc_int(&(cnt))
#endif

/* ======================================================================
 * GPU helper functions
 * ====================================================================== */

void OSHMPIU_gpu_init(void);
void OSHMPIU_gpu_finalize(void);

typedef enum {
    OSHMPIU_GPU_POINTER_UNREGISTERED_HOST,
    OSHMPIU_GPU_POINTER_REGISTERED_HOST,
    OSHMPIU_GPU_POINTER_DEV,
    OSHMPIU_GPU_POINTER_MANAGED
} OSHMPIU_gpu_pointer_type_t;

OSHMPI_STATIC_INLINE_PREFIX OSHMPIU_gpu_pointer_type_t OSHMPIU_gpu_query_pointer_type(const void
                                                                                      *ptr);

#include "utlist.h"
#include "thread.h"

#ifdef OSHMPI_ENABLE_CUDA
#include "gpu/cuda.h"
#elif defined(OSHMPI_ENABLE_ZE)
#include "gpu/ze.h"
#else
OSHMPI_STATIC_INLINE_PREFIX OSHMPIU_gpu_pointer_type_t OSHMPIU_gpu_query_pointer_type(const void
                                                                                      *ptr)
{
    return OSHMPIU_GPU_POINTER_UNREGISTERED_HOST;
}
#endif

#endif /* OSHMPI_UTIL_H */

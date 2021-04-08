/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef OSHMPI_IMPL_H
#define OSHMPI_IMPL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <oshmpiconf.h>
#if defined(OSHMPI_ENABLE_THREAD_MULTIPLE) || defined(OSHMPI_ENABLE_THREAD_SERIALIZED)
#include <pthread.h>
#endif
#include <shmem.h>
#include <shmemx.h>

#include "dlmalloc.h"
#include "oshmpi_util.h"
#include "am_pre.h"

#define OSHMPI_DEFAULT_SYMM_HEAP_SIZE (1L<<27)  /* 128MB */
#define OSHMPI_DEFAULT_DEBUG 0

/* DLMALLOC minimum allocated size (see create_mspace_with_base)
 * Part (less than 128*sizeof(size_t) bytes) of this space is used for bookkeeping,
 * so the capacity must be at least this large */
#define OSHMPI_DLMALLOC_MIN_MSPACE_SIZE (128 * sizeof(size_t))

#define OSHMPI_MPI_COLL32_T MPI_UINT32_T
#define OSHMPI_MPI_COLL64_T MPI_UINT64_T

#define OSHMPI_LOCK_MSG_TAG 999 /* For lock routines */

#define OSHMPI_DEFAULT_THREAD_SAFETY SHMEM_THREAD_SINGLE

typedef enum {
    OSHMPI_SOBJ_SYMM_DATA = 0,
    OSHMPI_SOBJ_SYMM_HEAP = 1,
    OSHMPI_SOBJ_SPACE_HEAP = 2,
    OSHMPI_SOBJ_SPACE_ATTACHED_HEAP = 3,
} OSHMPI_sobj_kind_t;

#define OSHMPI_SOBJ_HANDLE_KIND_MASK 0xc0000000
#define OSHMPI_SOBJ_HANDLE_KIND_SHIFT 30

#define OSHMPI_SOBJ_HANDLE_SYMMBIT_MASK 0x20000000
#define OSHMPI_SOBJ_HANDLE_SYMMBIT_SHIFT 29

#define OSHMPI_SOBJ_HANDLE_IDX_MASK 0x1FFFFFFF
#define OSHMPI_SOBJ_HANDLE_GET_KIND(handle) (((handle) & OSHMPI_SOBJ_HANDLE_KIND_MASK) >> OSHMPI_SOBJ_HANDLE_KIND_SHIFT)
#define OSHMPI_SOBJ_HANDLE_GET_SYMMBIT(handle) (((handle) & OSHMPI_SOBJ_HANDLE_SYMMBIT_MASK) >> OSHMPI_SOBJ_HANDLE_SYMMBIT_SHIFT)
#define OSHMPI_SOBJ_HANDLE_CHECK_SYMMBIT(handle) ((handle) & OSHMPI_SOBJ_HANDLE_SYMMBIT_MASK)
#define OSHMPI_SOBJ_HANDLE_GET_IDX(handle) ((handle) & OSHMPI_SOBJ_HANDLE_IDX_MASK)
#define OSHMPI_SOBJ_SET_HANDLE(kind, symm, idx) (((kind) << OSHMPI_SOBJ_HANDLE_KIND_SHIFT) | \
                                                ((symm) << OSHMPI_SOBJ_HANDLE_SYMMBIT_SHIFT) | (idx))

#if defined(OSHMPI_ENABLE_DIRECT_AMO)
#define OSHMPI_ENABLE_DIRECT_AMO_RUNTIME 1
#elif defined(OSHMPI_ENABLE_AM_AMO)
#define OSHMPI_ENABLE_DIRECT_AMO_RUNTIME 0
#else /* default make decision at runtime */

#ifdef OSHMPI_DISABLE_DEBUG
#define OSHMPI_ENABLE_DIRECT_AMO_RUNTIME (OSHMPI_global.amo_direct)
#else /* runtime with debug */
#define OSHMPI_ENABLE_DIRECT_AMO_RUNTIME (OSHMPI_env.amo_dbg_mode == OSHMPI_DBG_DIRECT ||   \
                                            (OSHMPI_env.amo_dbg_mode == OSHMPI_DBG_AUTO &&  \
                                                    OSHMPI_global.amo_direct))
#endif /* end of OSHMPI_DISABLE_DEBUG */
#endif /* end of OSHMPI_ENABLE_DIRECT_AMO */

typedef enum {
    OSHMPI_PUT,
    OSHMPI_GET,
} OSHMPI_rma_op_t;

OSHMPI_STATIC_INLINE_PREFIX int OSHMPI_check_gpu_direct_rma(const void *origin_addr,
                                                            shmemx_memkind_t sobj_memkind,
                                                            OSHMPI_rma_op_t rma);

#if defined(OSHMPI_ENABLE_AM_RMA)
#define OSHMPI_ENABLE_DIRECT_RMA_CONFIG 0
#define OSHMPI_ENABLE_DIRECT_RMA_RUNTIME(origin_addr, sobj_memkind, rma) 0
#elif defined(OSHMPI_ENABLE_DIRECT_RMA)
#define OSHMPI_ENABLE_DIRECT_RMA_CONFIG 1
#define OSHMPI_ENABLE_DIRECT_RMA_RUNTIME(origin_addr, sobj_memkind, rma) 1
#else /* default make decision at runtime */
#define OSHMPI_ENABLE_DIRECT_RMA_CONFIG 0

#ifdef OSHMPI_DISABLE_DEBUG
#define OSHMPI_ENABLE_DIRECT_RMA_RUNTIME(origin_addr, sobj_memkind, rma) \
            OSHMPI_check_gpu_direct_rma(origin_addr, sobj_memkind, rma)
#else /* runtime with debug */
#define OSHMPI_ENABLE_DIRECT_RMA_RUNTIME(origin_addr, sobj_memkind, rma) \
            (OSHMPI_env.rma_dbg_mode == OSHMPI_DBG_DIRECT ||             \
                (OSHMPI_env.rma_dbg_mode == OSHMPI_DBG_AUTO &&           \
                        OSHMPI_check_gpu_direct_rma(origin_addr, sobj_memkind, rma)))

#endif /* end of OSHMPI_DISABLE_DEBUG */
#endif /* end of OSHMPI_ENABLE_AM_RMA */

#if defined(OSHMPI_ENABLE_AM_ASYNC_THREAD)
#define OSHMPI_ENABLE_AM_ASYNC_THREAD_RUNTIME 1
#elif defined(OSHMPI_DISABLE_AM_ASYNC_THREAD)
#define OSHMPI_ENABLE_AM_ASYNC_THREAD_RUNTIME 0
#else /* default make decision at runtime */
#define OSHMPI_ENABLE_AM_ASYNC_THREAD_RUNTIME (OSHMPI_env.enable_async_thread)
#endif

typedef enum {
    OSHMPI_RELATIVE_DISP,
    OSHMPI_ABS_DISP
} OSHMPI_disp_mode_t;

#if defined(OSHMPI_ENABLE_DYNAMIC_WIN)
#define OSHMPI_ICTX_DISP_MODE(ictx) ((ictx)->disp_mode)
#else /* constant when dynamic window is disabled */
#define OSHMPI_ICTX_DISP_MODE(ictx) (OSHMPI_RELATIVE_DISP)
#endif

#if defined(OSHMPI_ENABLE_DYNAMIC_WIN)
#define OSHMPI_ICTX_SET_DISP_MODE(ictx, mode) do { (ictx)->disp_mode = (mode); } while (0)
#else /* no-op when dynamic window is disabled */
#define OSHMPI_ICTX_SET_DISP_MODE(ictx, mode) do {} while (0);
#endif

typedef struct OSHMPI_ictx {
    MPI_Win win;
    OSHMPIU_atomic_flag_t outstanding_op;
#if defined(OSHMPI_ENABLE_DYNAMIC_WIN)
    OSHMPI_disp_mode_t disp_mode;
#endif
} OSHMPI_ictx_t;

typedef struct OSHMPI_sobj_attr {
    uint32_t handle;
    shmemx_memkind_t memkind;
    void *base;
    MPI_Aint size;
#if defined(OSHMPI_ENABLE_DYNAMIC_WIN)
    MPI_Aint *base_offsets;     /* Array of offset of remote processes' base against local base.
                                 * Unused if (handle & symm_bit).*/
#endif
} OSHMPI_sobj_attr_t;

typedef struct OSHMPI_ctx {
    OSHMPI_ictx_t ictx;
    OSHMPIU_atomic_flag_t used_flag;
    OSHMPI_sobj_attr_t sobj_attr;
} OSHMPI_ctx_t;

typedef struct OSHMPI_space {
    OSHMPI_sobj_attr_t sobj_attr;
    OSUMPIU_mempool_t mem_pool;
    OSHMPIU_thread_cs_t mem_pool_cs;
#if !defined(OSHMPI_ENABLE_DYNAMIC_WIN) /* If dynamic win is enabled, attach to symm_ictx */
    OSHMPI_ictx_t default_ictx;
#endif
    OSHMPI_ctx_t *ctx_list;     /* contexts created for this space. */

    shmemx_space_config_t config;
    struct OSHMPI_space *next;
} OSHMPI_space_t;

typedef struct OSHMPI_space_list {
    OSHMPI_space_t *head;
    int nspaces;
    OSHMPIU_thread_cs_t cs;
} OSHMPI_space_list_t;

struct OSHMPI_am_pkt;

typedef struct OSHMPI_team {
    int my_pe;
    int n_pes;
    MPI_Comm comm;
    MPI_Group group;
    shmem_team_config_t config;
} OSHMPI_team_t;

typedef struct {
    int is_initialized;
    int is_start_pes_initialized;
    int team_world_my_pe;     /* cache of my_pe for SHMEM_TEAM_WORLD */
    int team_world_n_pes;     /* cache of n_pes for SHMEM_TEAM_WORLD */
    int thread_level;
    size_t page_sz;

    /* cache of comm, group for SHMEM_TEAM_WORLD */
    MPI_Comm team_world_comm;        /* duplicate of COMM_WORLD */
    MPI_Group team_world_group;
    OSHMPI_team_t *team_world;  /* cache a team object for easier code in split */

    /* cache of comm, group, my_pe, n_pes for SHMEM_TEAM_SHARED */
    MPI_Comm team_shared_comm;       /* shared split of COMM_WORLD */
    MPI_Group team_shared_group;
    int team_shared_my_pe;
    int team_shared_n_pes;
    OSHMPI_team_t *team_shared;  /* cache a team object for easier code in split */

    OSHMPI_sobj_attr_t symm_heap_attr;
    OSHMPI_sobj_attr_t symm_data_attr;

#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
    OSHMPI_ictx_t symm_ictx;
#else
    OSHMPI_ictx_t symm_heap_ictx;
    OSHMPI_ictx_t symm_data_ictx;
#endif

    MPI_Aint symm_heap_true_size;
    mspace symm_heap_mspace;
    OSHMPIU_thread_cs_t symm_heap_mspace_cs;

    OSHMPI_space_list_t space_list;

    unsigned int amo_direct;    /* Valid only when --enable-amo=runtime is set.
                                 * User may control it through env var
                                 * OSHMPI_AMO_OPS (see amo_ops in OSHMPI_env_t). */
} OSHMPI_global_t;

typedef enum {
    OSHMPI_AMO_CSWAP,
    OSHMPI_AMO_FINC,
    OSHMPI_AMO_INC,
    OSHMPI_AMO_FADD,
    OSHMPI_AMO_ADD,
    OSHMPI_AMO_FETCH,
    OSHMPI_AMO_SET,
    OSHMPI_AMO_SWAP,
    OSHMPI_AMO_FAND,
    OSHMPI_AMO_AND,
    OSHMPI_AMO_FOR,
    OSHMPI_AMO_OR,
    OSHMPI_AMO_FXOR,
    OSHMPI_AMO_XOR,
    OSHMPI_AMO_OP_LAST,
} OSHMPI_amo_op_shift_t;

typedef enum {
    OSHMPI_MPI_GPU_PT2PT,       /* MPI supports PT2PT with GPU buffers */
    OSHMPI_MPI_GPU_PUT,         /* MPI supports PUT with GPU buffers */
    OSHMPI_MPI_GPU_GET,         /* MPI supports GET with GPU buffers */
    OSHMPI_MPI_GPU_ACCUMULATES, /* MPI supports ACCUMULATES with GPU buffers */
} OSHMPI_mpi_gpu_feature_shift_t;

#define OSHMPI_CHECK_MPI_GPU_FEATURE(f) (OSHMPI_env.mpi_gpu_features & (1<<(f)))
#define OSHMPI_SET_MPI_GPU_FEATURE(f) (1<<(f))

typedef enum {
    OSHMPI_DBG_AUTO,
    OSHMPI_DBG_AM,
    OSHMPI_DBG_DIRECT,
} OSHMPI_dbg_mode_t;

typedef struct {
    /* SHMEM standard environment variables */
    MPI_Aint symm_heap_size;    /* SHMEM_SYMMETRIC_SIZE: Number of bytes to allocate for symmetric heap.
                                 * Value: Non-negative integer. Default OSHMPI_DEFAULT_SYMM_HEAP_SIZE. */
    unsigned int debug;         /* SHMEM_DEBUG: Enable debugging messages.
                                 * Value: 0 (default) |any non-zero value.
                                 * Always disabled when --enable-fast is set. */
    unsigned int version;       /* SHMEM_VERSION: Print the library version at start-up.
                                 * Value: 0 (default) |any non-zero value. */
    unsigned int info;          /* SHMEM_INFO: Print helpful text about all these environment variables.
                                 * Value: 0 (default) |any non-zero value. */

    /* OSHMPI extended environment variables */
    unsigned int verbose;       /* OSHMPI_VERBOSE: Print value of all OSHMPI configuration including
                                 * SHMEM standard environment varibales and OSHMPI extension.
                                 * Value: 0 (default) |any non-zero value. */
    uint32_t amo_ops;           /* OSHMPI_AMO_OPS: Arbitrary combination with bit shift defined in
                                 * OSHMPI_amo_op_shift_t. any_op and none are two special values.
                                 * any_op by default. */
    unsigned int enable_async_thread;   /* OSHMPI_ENABLE_ASYNC_THREAD:
                                         * Invalid when OSHMPI_DISABLE_AM_ASYNC_THREAD is set.
                                         * Default value is 1 if either AMO or RMA is AM based;
                                         * otherwise 0.*/
    uint32_t mpi_gpu_features;  /* OSHMPI_MPI_GPU_FEATURES: Arbitrary combination with bit shift defined in
                                 * OSHMPI_mpi_gpu_feature_shift_t. none and all are two special values. */
#ifndef OSHMPI_DISABLE_DEBUG
    OSHMPI_dbg_mode_t amo_dbg_mode;
    OSHMPI_dbg_mode_t rma_dbg_mode;
#endif
} OSHMPI_env_t;

#ifdef OSHMPI_ENABLE_IPO        /* define empty bracket to be compatible with code cleanup script */
#define OSHMPI_FORCEINLINE() _Pragma("forceinline")
#define OSHMPI_NOINLINE_RECURSIVE() _Pragma("noinline recursive")
#else
#define OSHMPI_FORCEINLINE() ;
#define OSHMPI_NOINLINE_RECURSIVE() ;
#endif

extern OSHMPI_global_t OSHMPI_global;
extern OSHMPI_env_t OSHMPI_env;

/* Per-object critical section MACROs. */
#ifdef OSHMPI_ENABLE_THREAD_MULTIPLE
#define OSHMPI_THREAD_INIT_CS(cs_ptr)  do {                   \
    if (OSHMPI_global.thread_level == SHMEM_THREAD_MULTIPLE) {\
        int __err OSHMPI_ATTRIBUTE((unused));                 \
        __err = OSHMPIU_thread_cs_init(cs_ptr);               \
        OSHMPI_ASSERT(!__err);                                \
    }                                                         \
} while (0)

#define OSHMPI_THREAD_DESTROY_CS(cs_ptr)  do {                 \
    if (OSHMPI_global.thread_level == SHMEM_THREAD_MULTIPLE && \
            OSHMPIU_THREAD_CS_IS_INITIALIZED(cs_ptr)) {        \
        int __err OSHMPI_ATTRIBUTE((unused));                  \
        __err = OSHMPIU_thread_cs_destroy(cs_ptr);             \
        OSHMPI_ASSERT(!__err);                                 \
    }                                                          \
} while (0)

#define OSHMPI_THREAD_ENTER_CS(cs_ptr)  do {                          \
    if (OSHMPI_global.thread_level == SHMEM_THREAD_MULTIPLE) {        \
        int __err OSHMPI_ATTRIBUTE((unused));                         \
        __err = OSHMPIU_THREAD_CS_ENTER(cs_ptr);                      \
        OSHMPI_ASSERT(!__err);                                        \
    }                                                                 \
} while (0)

#define OSHMPI_THREAD_EXIT_CS(cs_ptr)  do {                     \
    if (OSHMPI_global.thread_level == SHMEM_THREAD_MULTIPLE) {  \
        int __err OSHMPI_ATTRIBUTE((unused));                   \
        __err = OSHMPIU_THREAD_CS_EXIT(cs_ptr);                 \
        OSHMPI_ASSERT(!__err);                                  \
    }                                                           \
} while (0)

#else /* OSHMPI_ENABLE_THREAD_MULTIPLE */
#define OSHMPI_THREAD_INIT_CS(cs_ptr)
#define OSHMPI_THREAD_DESTROY_CS(cs_ptr)
#define OSHMPI_THREAD_ENTER_CS(cs_ptr)
#define OSHMPI_THREAD_EXIT_CS(cs_ptr)
#endif /* OSHMPI_ENABLE_THREAD_MULTIPLE */

#define OSHMPI_TEAM_HANDLE_TO_OBJ(handle) ((OSHMPI_team_t *) (handle))
#define OSHMPI_TEAM_OBJ_TO_HANDLE(obj) ((shmem_team_t) (obj))

/* SHMEM internal routines. */
void OSHMPI_initialize_thread(int required, int *provided);
void OSHMPI_implicit_finalize(void);
void OSHMPI_finalize(void);
void OSHMPI_global_exit(int status);
void OSHMPI_set_mpi_info_args(MPI_Info info);

void *OSHMPI_malloc(size_t size);
void OSHMPI_free(void *ptr);
void *OSHMPI_realloc(void *ptr, size_t size);
void *OSHMPI_align(size_t alignment, size_t size);

void OSHMPI_strided_initialize(void);
void OSHMPI_strided_finalize(void);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_create_strided_dtype(size_t nelems, ptrdiff_t stride,
                                                             MPI_Datatype mpi_type,
                                                             size_t required_ext_nelems,
                                                             size_t * strided_cnt,
                                                             MPI_Datatype * strided_type);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_free_strided_dtype(MPI_Datatype mpi_type,
                                                           MPI_Datatype * strided_type);

void OSHMPI_space_initialize(void);
void OSHMPI_space_finalize(void);
void OSHMPI_space_create(shmemx_space_config_t space_config, OSHMPI_space_t ** space_ptr);
void OSHMPI_space_destroy(OSHMPI_space_t * space);
int OSHMPI_space_create_ctx(OSHMPI_space_t * space, long options, OSHMPI_ctx_t ** ctx_ptr);
void OSHMPI_space_attach(OSHMPI_space_t * space);
void OSHMPI_space_detach(OSHMPI_space_t * space);
void *OSHMPI_space_malloc(OSHMPI_space_t * space, size_t size);
void *OSHMPI_space_align(OSHMPI_space_t * space, size_t alignment, size_t size);
void OSHMPI_space_free(OSHMPI_space_t * space, void *ptr);

void OSHMPI_ctx_destroy(OSHMPI_ctx_t * ctx);

int OSHMPI_team_create(OSHMPI_team_t ** team);
void OSHMPI_team_destroy(OSHMPI_team_t ** team);

/* Subroutines for rma. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_put_nbi(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                    MPI_Datatype mpi_type, size_t typesz,
                                                    const void *origin_addr, void *target_addr,
                                                    size_t nelems, int pe);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_put(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                MPI_Datatype mpi_type, size_t typesz,
                                                const void *origin_addr, void *target_addr,
                                                size_t nelems, int pe);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_iput(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 MPI_Datatype mpi_type,
                                                 OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                 const void *origin_addr, void *target_addr,
                                                 ptrdiff_t target_st, ptrdiff_t origin_st,
                                                 size_t nelems, int pe);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_get_nbi(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                    MPI_Datatype mpi_type, size_t typesz,
                                                    void *origin_addr, const void *target_addr,
                                                    size_t nelems, int pe);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_get(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                MPI_Datatype mpi_type, size_t typesz,
                                                void *origin_addr, const void *target_addr,
                                                size_t nelems, int pe);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_iget(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 MPI_Datatype mpi_type,
                                                 OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                 void *origin_addr, const void *target_addr,
                                                 ptrdiff_t origin_st, ptrdiff_t target_st,
                                                 size_t nelems, int pe);

/* Subroutines for am rma. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_rma_am_put(OSHMPI_ictx_t * ictx,
                                                   MPI_Datatype mpi_type, size_t typesz,
                                                   const void *origin_addr, void *target_addr,
                                                   size_t nelems, int pe,
                                                   OSHMPI_sobj_attr_t * sobj_attr);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_rma_am_get(OSHMPI_ictx_t * ictx, MPI_Datatype mpi_type,
                                                   size_t typesz, void *origin_addr,
                                                   const void *target_addr, size_t nelems, int pe,
                                                   OSHMPI_sobj_attr_t * sobj_attr);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_rma_am_iput(OSHMPI_ictx_t * ictx, MPI_Datatype mpi_type,
                                                    OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                    const void *origin_addr, void *target_addr,
                                                    ptrdiff_t origin_st, ptrdiff_t target_st,
                                                    size_t nelems, int pe,
                                                    OSHMPI_sobj_attr_t * sobj_attr);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_rma_am_iget(OSHMPI_ictx_t * ictx, MPI_Datatype mpi_type,
                                                    OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                    void *origin_addr, const void *target_addr,
                                                    ptrdiff_t origin_st, ptrdiff_t target_st,
                                                    size_t nelems, int pe,
                                                    OSHMPI_sobj_attr_t * sobj_attr);

void OSHMPI_rma_am_initialize(void);
void OSHMPI_rma_am_finalize(void);
void OSHMPI_rma_am_put_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt);
void OSHMPI_rma_am_get_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt);
void OSHMPI_rma_am_iput_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt);
void OSHMPI_rma_am_iget_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt);

/* Subroutines for collectives. */
void OSHMPI_coll_initialize(void);
void OSHMPI_coll_finalize(void);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_barrier_all(void);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_barrier(int PE_start, int logPE_stride, int PE_size);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_sync_all(void);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_sync(int PE_start, int logPE_stride, int PE_size);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_broadcast(void *dest, const void *source, size_t nelems,
                                                  MPI_Datatype mpi_type, int PE_root, int PE_start,
                                                  int logPE_stride, int PE_size);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_collect(void *dest, const void *source, size_t nelems,
                                                MPI_Datatype mpi_type, int PE_start,
                                                int logPE_stride, int PE_size);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_fcollect(void *dest, const void *source, size_t nelems,
                                                 MPI_Datatype mpi_type, int PE_start,
                                                 int logPE_stride, int PE_size);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_alltoall(void *dest, const void *source, size_t nelems,
                                                 MPI_Datatype mpi_type, int PE_start,
                                                 int logPE_stride, int PE_size);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_alltoalls(void *dest, const void *source, ptrdiff_t dst,
                                                  ptrdiff_t sst, size_t nelems,
                                                  MPI_Datatype mpi_type, int PE_start,
                                                  int logPE_stride, int PE_size);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_allreduce(void *dest, const void *source, int count,
                                                  MPI_Datatype mpi_type, MPI_Op op, int PE_start,
                                                  int logPE_stride, int PE_size);

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_fence(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)));
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_quiet(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)));

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_set_lock(long *lockp);
OSHMPI_STATIC_INLINE_PREFIX int OSHMPI_test_lock(long *lockp);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_clear_lock(long *lockp);

/* Subroutines for am routines. */
void OSHMPI_am_initialize(void);
void OSHMPI_am_finalize(void);
void OSHMPI_am_cb_progress(void);
void OSHMPI_am_cb_regist(OSHMPI_am_pkt_type_t pkt_type, const char *pkt_name,
                         OSHMPI_am_cb_t cb_func);
OSHMPI_STATIC_INLINE_PREFIX int OSHMPI_am_get_pkt_ptag(void);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_flush(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 int PE_start, int logPE_stride, int PE_size);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_flush_all(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)));
void OSHMPI_am_flush_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt);

/* Subroutines for atomics. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_cswap(shmem_ctx_t ctx
                                                  OSHMPI_ATTRIBUTE((unused)), MPI_Datatype mpi_type,
                                                  OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                  size_t bytes, void *dest, void *cond_ptr,
                                                  void *value_ptr, int pe, void *oldval_ptr);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_fetch(shmem_ctx_t ctx
                                                  OSHMPI_ATTRIBUTE((unused)), MPI_Datatype mpi_type,
                                                  OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                  size_t bytes, MPI_Op op,
                                                  OSHMPI_am_mpi_op_index_t op_idx, void *dest,
                                                  void *value_ptr, int pe, void *oldval_ptr);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_post(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 MPI_Datatype mpi_type,
                                                 OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                 size_t bytes, MPI_Op op,
                                                 OSHMPI_am_mpi_op_index_t op_idx, void *dest,
                                                 void *value_ptr, int pe);

/* Subroutines for am atomics. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_cswap(OSHMPI_ictx_t * ictx,
                                                     MPI_Datatype mpi_type,
                                                     OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                     size_t bytes, void *dest, void *cond_ptr,
                                                     void *value_ptr, int pe, void *oldval_ptr,
                                                     OSHMPI_sobj_attr_t * sobj_attr);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_fetch(OSHMPI_ictx_t * ictx,
                                                     MPI_Datatype mpi_type,
                                                     OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                     size_t bytes, MPI_Op op,
                                                     OSHMPI_am_mpi_op_index_t op_idx, void *dest,
                                                     void *value_ptr, int pe, void *oldval_ptr,
                                                     OSHMPI_sobj_attr_t * sobj_attr);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_post(OSHMPI_ictx_t * ictx,
                                                    MPI_Datatype mpi_type,
                                                    OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                    size_t bytes, MPI_Op op,
                                                    OSHMPI_am_mpi_op_index_t op_idx, void *dest,
                                                    void *value_ptr, int pe,
                                                    OSHMPI_sobj_attr_t * sobj_attr);

void OSHMPI_amo_am_initialize(void);
void OSHMPI_amo_am_finalize(void);
void OSHMPI_amo_am_cswap_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt);
void OSHMPI_amo_am_fetch_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt);
void OSHMPI_amo_am_post_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt);

/* Wrapper of MPI blocking calls with active message progress. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_send(const void *buf, int count,
                                                             MPI_Datatype datatype, int dest,
                                                             int tag, MPI_Comm comm);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_recv(void *buf, int count,
                                                             MPI_Datatype datatype, int src,
                                                             int tag, MPI_Comm comm,
                                                             MPI_Status * status);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_waitall(int count,
                                                                MPI_Request array_of_requests[],
                                                                MPI_Status array_of_statuses[]);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_barrier(MPI_Comm comm);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_bcast(void *buffer, int count,
                                                              MPI_Datatype datatype, int root,
                                                              MPI_Comm comm);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_allgather(const void *sendbuf,
                                                                  int sendcount,
                                                                  MPI_Datatype sendtype,
                                                                  void *recvbuf, int recvcount,
                                                                  MPI_Datatype recvtype,
                                                                  MPI_Comm comm);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_allgatherv(const void *sendbuf,
                                                                   int sendcount,
                                                                   MPI_Datatype sendtype,
                                                                   void *recvbuf,
                                                                   const int *recvcounts,
                                                                   const int *displs,
                                                                   MPI_Datatype recvtype,
                                                                   MPI_Comm comm);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_alltoall(const void *sendbuf,
                                                                 int sendcount,
                                                                 MPI_Datatype sendtype,
                                                                 void *recvbuf, int recvcount,
                                                                 MPI_Datatype recvtype,
                                                                 MPI_Comm comm);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_allreduce(const void *sendbuf,
                                                                  void *recvbuf, int count,
                                                                  MPI_Datatype datatype,
                                                                  MPI_Op op, MPI_Comm comm);

/* Common routines for symm objects (symm heap, data, space heap) */
void OSHMPI_sobj_init_attr(OSHMPI_sobj_attr_t * sobj_attr, shmemx_memkind_t memkind,
                           void *base, MPI_Aint size);
void OSHMPI_sobj_symm_info_allgather(OSHMPI_sobj_attr_t * sobj_attr, int *symm_flag);
void OSHMPI_sobj_destroy_attr(OSHMPI_sobj_attr_t * sobj_attr);
void OSHMPI_sobj_symm_info_dbgprint(OSHMPI_sobj_attr_t * sobj_attr);

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_sobj_set_handle(OSHMPI_sobj_attr_t * sobj_attr,
                                                        OSHMPI_sobj_kind_t sobj_kind, int symm_flag,
                                                        int idx)
{
    sobj_attr->handle = OSHMPI_SOBJ_SET_HANDLE(sobj_kind, symm_flag, idx);
}

OSHMPI_STATIC_INLINE_PREFIX int OSHMPI_sobj_check_range(const void *ptr,
                                                        OSHMPI_sobj_attr_t sobj_attr)
{
    return ((MPI_Aint) ptr >= (MPI_Aint) sobj_attr.base
            && (MPI_Aint) ptr < (MPI_Aint) sobj_attr.base + sobj_attr.size) ? 1 : 0;
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_sobj_trans_vaddr_to_disp(OSHMPI_sobj_attr_t * sobj_attr,
                                                                 const void *abs_addr,
                                                                 int target_rank, int disp_mode,
                                                                 MPI_Aint * disp_ptr)
{
#if defined(OSHMPI_ENABLE_DYNAMIC_WIN)
    if (disp_mode == OSHMPI_ABS_DISP) {
        if (OSHMPI_SOBJ_HANDLE_CHECK_SYMMBIT(sobj_attr->handle)) {
            /* symmetric heap, local absolute addr is equal to remote */
            OSHMPI_FORCEINLINE()
                OSHMPI_CALLMPI(MPI_Get_address(abs_addr, disp_ptr));
        } else {
            /* asymmetric heap, compute remote absolute addr */
            MPI_Aint abs_disp = 0;
            OSHMPI_FORCEINLINE()
                OSHMPI_CALLMPI(MPI_Get_address(abs_addr, &abs_disp));
            OSHMPI_FORCEINLINE()
                OSHMPI_CALLMPI(*disp_ptr = MPI_Aint_add(abs_disp,
                                                        sobj_attr->base_offsets[target_rank]));
        }
    } else
#endif
    {   /* Compute relative displacement */
        MPI_Aint abs_disp = 0, base_disp = 0;
        OSHMPI_FORCEINLINE()
            OSHMPI_CALLMPI(MPI_Get_address(abs_addr, &abs_disp));
        OSHMPI_FORCEINLINE()
            OSHMPI_CALLMPI(MPI_Get_address(sobj_attr->base, &base_disp));
        OSHMPI_FORCEINLINE()
            OSHMPI_CALLMPI(*disp_ptr = MPI_Aint_diff(abs_disp, base_disp));
    }
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_sobj_query_attr_ictx(OSHMPI_ctx_t * ctx,
                                                             const void *abs_addr,
                                                             int target_rank,
                                                             OSHMPI_sobj_attr_t ** sobj_attr_ptr,
                                                             OSHMPI_ictx_t ** ictx_ptr)
{
    if (ctx != SHMEM_CTX_DEFAULT) {
        *ictx_ptr = &ctx->ictx;
        *sobj_attr_ptr = &ctx->sobj_attr;
        return;
    }

    /* search default contexts */
    if (OSHMPI_sobj_check_range(abs_addr, OSHMPI_global.symm_heap_attr)) {
#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
        *ictx_ptr = &OSHMPI_global.symm_ictx;
#else
        *ictx_ptr = &OSHMPI_global.symm_heap_ictx;
#endif
        *sobj_attr_ptr = &OSHMPI_global.symm_heap_attr;
        return;
    }

    if (OSHMPI_sobj_check_range(abs_addr, OSHMPI_global.symm_data_attr)) {
#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
        *ictx_ptr = &OSHMPI_global.symm_ictx;
#else
        *ictx_ptr = &OSHMPI_global.symm_data_ictx;
#endif
        *sobj_attr_ptr = &OSHMPI_global.symm_data_attr;
        return;
    }

    /* Search spaces */
    OSHMPI_space_t *space, *tmp;
    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.space_list.cs);
    LL_FOREACH_SAFE(OSHMPI_global.space_list.head, space, tmp) {
        if (OSHMPI_sobj_check_range(abs_addr, space->sobj_attr)) {
#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
            *ictx_ptr = &OSHMPI_global.symm_ictx;
#else
            *ictx_ptr = &space->default_ictx;
#endif
            *sobj_attr_ptr = &space->sobj_attr;
            break;
        }
    }
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.space_list.cs);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_sobj_trans_disp_to_vaddr(uint32_t sobj_handle,
                                                                 MPI_Aint disp, void **vaddr)
{
    OSHMPI_space_t *space, *tmp;

    /* AM based routines always use relative displacement */

    switch (OSHMPI_SOBJ_HANDLE_GET_KIND(sobj_handle)) {
        case OSHMPI_SOBJ_SYMM_HEAP:
            *vaddr = (void *) ((char *) OSHMPI_global.symm_heap_attr.base + disp);
            break;
        case OSHMPI_SOBJ_SYMM_DATA:
            *vaddr = (void *) ((char *) OSHMPI_global.symm_data_attr.base + disp);
            break;
        case OSHMPI_SOBJ_SPACE_ATTACHED_HEAP:
            /* Search spaces */
            OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.space_list.cs);
            LL_FOREACH_SAFE(OSHMPI_global.space_list.head, space, tmp) {
                if (space->sobj_attr.handle == sobj_handle) {
                    *vaddr = (void *) ((char *) space->sobj_attr.base + disp);
                    break;
                }
            }
            OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.space_list.cs);
            break;
        case OSHMPI_SOBJ_SPACE_HEAP:
        default:
            OSHMPI_ERR_ABORT("Unsupported symmetric object kind:%d, handle 0x%x\n",
                             OSHMPI_SOBJ_HANDLE_GET_KIND(sobj_handle), sobj_handle);
            break;
    }
}

/* Common routines for internal use */
OSHMPI_STATIC_INLINE_PREFIX int OSHMPI_check_gpu_direct_rma(const void *origin_addr,
                                                            shmemx_memkind_t sobj_memkind,
                                                            OSHMPI_rma_op_t rma)
{
    int use_gpu = 0;

    if (sobj_memkind == SHMEMX_MEM_CUDA || sobj_memkind == SHMEMX_MEM_ZE
        || OSHMPIU_gpu_query_pointer_type(origin_addr) == OSHMPIU_GPU_POINTER_DEV)
        use_gpu = 1;

    if (!use_gpu)
        return 1;       /* always direct for host buffers */

    int direct = 0;
    switch (rma) {
        case OSHMPI_PUT:
            direct = OSHMPI_CHECK_MPI_GPU_FEATURE(OSHMPI_MPI_GPU_PUT);
            break;
        case OSHMPI_GET:
            direct = OSHMPI_CHECK_MPI_GPU_FEATURE(OSHMPI_MPI_GPU_GET);
            break;
        default:
            OSHMPI_ASSERT(rma == OSHMPI_PUT || rma == OSHMPI_GET);
            break;
    }

    if (!direct && !OSHMPI_CHECK_MPI_GPU_FEATURE(OSHMPI_MPI_GPU_PT2PT)) {
        /* abort if GPU is used but MPI supports neither RMA nor P2P */
        OSHMPI_ERR_ABORT("MPI does not support GPU over RMA nor PT2PT."
                         "Set environment variable OSHMPI_MPI_GPU_FEATURES from \"pt2pt,put,get,acc\".\n");
        return 0;
    }

    return direct;
}

OSHMPI_STATIC_INLINE_PREFIX void ctx_local_complete_impl(int pe, OSHMPI_ictx_t * ictx)
{
    OSHMPI_FORCEINLINE()
        OSHMPI_CALLMPI(MPI_Win_flush_local(pe, ictx->win));
}


/* Workaround: some MPI routines may skip internal progress (e.g., MPICH CH3,
 * fetch_and_op(self) + flush_local(self) in test and wait_until). Thus,
 * we have to manually poll MPI progress at some "risky" MPI calls.  */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_progress_poll_mpi(void)
{
    int iprobe_flag = 0;

    /* No need to make manual MPI progress if we are making OSHMPI progress for AM AMO
     * by either main thread or asynchronous thread. */
#ifdef OSHMPI_ENABLE_AM_AMO
    return;
#elif !defined(OSHMPI_ENABLE_DIRECT_AMO)        /* auto */
    if (!OSHMPI_global.amo_direct)
        return;
#endif

    OSHMPI_CALLMPI(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, OSHMPI_global.team_world_comm,
                              &iprobe_flag, MPI_STATUS_IGNORE));
}

enum {
    OSHMPI_OP_OUTSTANDING,      /* nonblocking or PUT with local completion */
    OSHMPI_OP_COMPLETED         /* GET with local completion */
};


#ifdef OSHMPI_ENABLE_OP_TRACKING
#define OSHMPI_SET_OUTSTANDING_OP(ctx, completion) do {       \
        if (completion == OSHMPI_OP_COMPLETED) break;         \
        OSHMPIU_ATOMIC_FLAG_STORE(ctx->outstanding_op, 1);    \
        } while (0)
#else
#define OSHMPI_SET_OUTSTANDING_OP(ctx, completion) do {} while (0)
#endif


OSHMPI_STATIC_INLINE_PREFIX size_t OSHMPI_get_mspace_sz(size_t bufsz)
{
    size_t mspace_sz = 0;

    /* Ensure extra bookkeeping space in MSPACE */
    mspace_sz = bufsz + OSHMPI_DLMALLOC_MIN_MSPACE_SIZE;
    mspace_sz = OSHMPI_ALIGN(bufsz, OSHMPI_global.page_sz);

    return mspace_sz;
}

#include "strided_impl.h"
#include "coll_impl.h"
#include "rma_impl.h"
#include "amo_impl.h"
#include "am_impl.h"
#include "order_impl.h"
#include "p2p_impl.h"
#include "lock_impl.h"
#include "am_progress_impl.h"

#endif /* OSHMPI_IMPL_H */

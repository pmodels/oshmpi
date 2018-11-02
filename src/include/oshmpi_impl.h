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
#ifdef OSHMPI_ENABLE_AMO_ASYNC_THREAD
#include <pthread.h>
#endif
#include <shmem.h>

#include "dlmalloc.h"
#include "oshmpi_util.h"

#define OSHMPI_DEFAULT_SYMM_HEAP_SIZE (1L<<31)  /* 2GB */
#define OSHMPI_DEFAULT_DEBUG 0

/* DLMALLOC minimum allocated size (see create_mspace_with_base)
 * Part (less than 128*sizeof(size_t) bytes) of this space is used for bookkeeping,
 * so the capacity must be at least this large */
#define OSHMPI_DLMALLOC_MIN_MSPACE_SIZE (128 * sizeof(size_t))

#define OSHMPI_MPI_COLL32_T MPI_UINT32_T
#define OSHMPI_MPI_COLL64_T MPI_UINT64_T

#define OSHMPI_LOCK_MSG_TAG 999 /* For lock routines */

typedef struct OSHMPI_comm_cache_obj {
    int pe_start;
    int pe_stride;
    int pe_size;
    MPI_Comm comm;
    MPI_Group group;            /* Cached in case we need to translate root rank. */
    struct OSHMPI_comm_cache_obj *next;
} OSHMPI_comm_cache_obj_t;

typedef struct OSHMPI_comm_cache_list {
    OSHMPI_comm_cache_obj_t *head;
    int nobjs;
} OSHMPI_comm_cache_list_t;

struct OSHMPI_amo_pkt;

typedef struct {
    int is_initialized;
    int is_start_pes_initialized;
    int world_rank;
    int world_size;
    int thread_level;

    MPI_Comm comm_world;        /* duplicate of COMM_WORLD */
    MPI_Group comm_world_group;

    MPI_Win symm_heap_win;
    void *symm_heap_base;
    MPI_Aint symm_heap_size;
    mspace symm_heap_mspace;

    MPI_Win symm_data_win;
    void *symm_data_base;
    MPI_Aint symm_data_size;

    OSHMPI_comm_cache_list_t comm_cache_list;

#ifndef OSHMPI_ENABLE_DIRECT_AMO
    /* Active message based AMO */
    MPI_Comm amo_comm_world;    /* duplicate of COMM_WORLD, used for packet */
    MPI_Comm amo_ack_comm_world;        /* duplicate of COMM_WORLD, used for packet ACK */
#ifdef OSHMPI_ENABLE_AMO_ASYNC_THREAD
    pthread_mutex_t amo_async_mutex;
    pthread_cond_t amo_async_cond;
    volatile int amo_async_thread_done;
    pthread_t amo_async_thread;
#endif
    unsigned int *amo_outstanding_op_flags;     /* flag indicating whether outstanding AM
                                                 * based AMO exists. When a post AMO (nonblocking)
                                                 * has been issued, this flag becomes 1; when
                                                 * a flush or fetch/cswap AMO issued, reset to 0;
                                                 * We only need flush a remote PE when flag is 1.*/
    MPI_Request amo_req;
    struct OSHMPI_amo_pkt *amo_pkt;     /* Temporary pkt for receiving incoming active message.
                                         * Type OSHMPI_amo_pkt_t is loaded later than global struct,
                                         * thus keep it as pointer. */
    MPI_Datatype *amo_datatypes_table;
    MPI_Op *amo_ops_table;
#endif
} OSHMPI_global_t;

typedef struct {
    MPI_Aint symm_heap_size;    /* SHMEM_SYMMETRIC_SIZE */
    int debug;                  /*SHMEM_DEBUG, value: 0|1 */
} OSHMPI_env_t;

typedef enum {
    OSHMPI_SYMM_OBJ_HEAP,
    OSHMPI_SYMM_OBJ_DATA,
} OSHMPI_symm_obj_type_t;

/* Define AMO's MPI datatype indexes. Used when transfer datatype in AM packets.
 * Instead of using MPI datatypes directly, integer indexs can be safely
 * handled by switch structure.*/
typedef enum {
    OSHMPI_AMO_MPI_INT,
    OSHMPI_AMO_MPI_LONG,
    OSHMPI_AMO_MPI_LONG_LONG,
    OSHMPI_AMO_MPI_UNSIGNED,
    OSHMPI_AMO_MPI_UNSIGNED_LONG,
    OSHMPI_AMO_MPI_UNSIGNED_LONG_LONG,
    OSHMPI_AMO_MPI_INT32_T,
    OSHMPI_AMO_MPI_INT64_T,
    OSHMPI_AMO_MPI_UINT32_T,
    OSHMPI_AMO_MPI_UINT64_T,
    OSHMPI_AMO_OSHMPI_MPI_SIZE_T,
    OSHMPI_AMO_OSHMPI_MPI_PTRDIFF_T,
    OSHMPI_AMO_MPI_FLOAT,
    OSHMPI_AMO_MPI_DOUBLE,
    OSHMPI_AMO_MPI_DATATYPE_MAX,
} OSHMPI_amo_mpi_datatype_index_t;

typedef enum {
    OSHMPI_AMO_MPI_BAND,
    OSHMPI_AMO_MPI_BOR,
    OSHMPI_AMO_MPI_BXOR,
    OSHMPI_AMO_MPI_NO_OP,
    OSHMPI_AMO_MPI_REPLACE,
    OSHMPI_AMO_MPI_SUM,
    OSHMPI_AMO_MPI_OP_MAX,
} OSHMPI_amo_mpi_op_index_t;

extern OSHMPI_global_t OSHMPI_global;
extern OSHMPI_env_t OSHMPI_env;

int OSHMPI_initialize_thread(int required, int *provided);
void OSHMPI_implicit_finalize(void);
int OSHMPI_finalize(void);

OSHMPI_STATIC_INLINE_PREFIX void *OSHMPI_malloc(size_t size);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_free(void *ptr);
OSHMPI_STATIC_INLINE_PREFIX void *OSHMPI_realloc(void *ptr, size_t size);
OSHMPI_STATIC_INLINE_PREFIX void *OSHMPI_align(size_t alignment, size_t size);

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_put_nbi(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                    MPI_Datatype mpi_type, const void *origin_addr,
                                                    void *target_addr, size_t nelems, int pe);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_put(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                MPI_Datatype mpi_type, const void *origin_addr,
                                                void *target_addr, size_t nelems, int pe);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_iput(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 MPI_Datatype mpi_type, const void *origin_addr,
                                                 void *target_addr, ptrdiff_t target_st,
                                                 ptrdiff_t origin_st, size_t nelems, int pe);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_get_nbi(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                    MPI_Datatype mpi_type, void *origin_addr,
                                                    const void *target_addr, size_t nelems, int pe);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_get(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                MPI_Datatype mpi_type, void *origin_addr,
                                                const void *target_addr, size_t nelems, int pe);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_ctx_iget(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 MPI_Datatype mpi_type, void *origin_addr,
                                                 const void *target_addr, ptrdiff_t origin_st,
                                                 ptrdiff_t target_st, size_t nelems, int pe);

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_coll_initialize(void);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_coll_finalize(void);
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

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_initialize(void);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_finalize(void);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_cb_progress(void);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_cswap(shmem_ctx_t ctx
                                                  OSHMPI_ATTRIBUTE((unused)), MPI_Datatype mpi_type,
                                                  OSHMPI_amo_mpi_datatype_index_t mpi_type_idx,
                                                  size_t bytes, void *dest, void *cond_ptr,
                                                  void *value_ptr, int pe, void *oldval_ptr);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_fetch(shmem_ctx_t ctx
                                                  OSHMPI_ATTRIBUTE((unused)), MPI_Datatype mpi_type,
                                                  OSHMPI_amo_mpi_datatype_index_t mpi_type_idx,
                                                  size_t bytes, MPI_Op op,
                                                  OSHMPI_amo_mpi_op_index_t op_idx, void *dest,
                                                  void *value_ptr, int pe, void *oldval_ptr);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_post(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 MPI_Datatype mpi_type,
                                                 OSHMPI_amo_mpi_datatype_index_t mpi_type_idx,
                                                 size_t bytes, MPI_Op op,
                                                 OSHMPI_amo_mpi_op_index_t op_idx, void *dest,
                                                 void *value_ptr, int pe);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_flush(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                     int PE_start, int logPE_stride, int PE_size);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_flush_all(shmem_ctx_t ctx);

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

/* Common routines for internal use */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_translate_win_and_disp(const void *abs_addr,
                                                               MPI_Win * win_ptr,
                                                               MPI_Aint * disp_ptr)
{
    if (OSHMPI_global.symm_heap_base <= abs_addr &&
        (MPI_Aint) abs_addr <= (MPI_Aint) OSHMPI_global.symm_heap_base +
        OSHMPI_global.symm_heap_size) {
        /* heap */
        *disp_ptr = (MPI_Aint) abs_addr - (MPI_Aint) OSHMPI_global.symm_heap_base;
        *win_ptr = OSHMPI_global.symm_heap_win;
        return;
    } else if (OSHMPI_global.symm_data_base <= abs_addr &&
               (MPI_Aint) abs_addr <= (MPI_Aint) OSHMPI_global.symm_data_base +
               OSHMPI_global.symm_data_size) {
        /* text */
        *disp_ptr = (MPI_Aint) abs_addr - (MPI_Aint) OSHMPI_global.symm_data_base;
        *win_ptr = OSHMPI_global.symm_data_win;
    }
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_translate_disp_to_vaddr(OSHMPI_symm_obj_type_t symm_type,
                                                                int disp, void **vaddr)
{
    switch (symm_type) {
        case OSHMPI_SYMM_OBJ_HEAP:
            *vaddr = (void *) ((char *) OSHMPI_global.symm_heap_base + disp);
            break;
        case OSHMPI_SYMM_OBJ_DATA:
            *vaddr = (void *) ((char *) OSHMPI_global.symm_data_base + disp);
            break;
    }
}

/* Create derived datatype for strided data format.
 * If it is contig (stride == 1), then the basic datatype is returned.
 * The caller must check the returned datatype to free it when necessary. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_create_strided_dtype(size_t nelems, ptrdiff_t stride,
                                                             MPI_Datatype mpi_type,
                                                             size_t required_ext_nelems,
                                                             size_t * strided_cnt,
                                                             MPI_Datatype * strided_type)
{
    /* TODO: check non-int inputs exceeds int limit */

    if (stride == 1) {
        *strided_type = mpi_type;
        *strided_cnt = nelems;
    } else {
        MPI_Datatype vtype = MPI_DATATYPE_NULL;
        size_t elem_bytes = 0;

        OSHMPI_CALLMPI(MPI_Type_vector((int) nelems, 1, (int) stride, mpi_type, &vtype));

        /* Vector does not count stride after last chunk, thus we need to resize to
         * cover it when multiple elements with the stride_datatype may be used (i.e., alltoalls).
         * Extent can be negative in MPI, however, we do not expect such case in OSHMPI.
         * Thus skip any negative one */
        if (required_ext_nelems > 0) {
            if (mpi_type == OSHMPI_MPI_COLL32_T)
                elem_bytes = 4;
            else
                elem_bytes = 8;
            OSHMPI_CALLMPI(MPI_Type_create_resized
                           (vtype, 0, required_ext_nelems * elem_bytes, strided_type));
        } else
            *strided_type = vtype;
        OSHMPI_CALLMPI(MPI_Type_commit(strided_type));
        if (required_ext_nelems > 0)
            OSHMPI_CALLMPI(MPI_Type_free(&vtype));
        *strided_cnt = 1;
    }
}

OSHMPI_STATIC_INLINE_PREFIX void ctx_local_complete_impl(shmem_ctx_t ctx
                                                         OSHMPI_ATTRIBUTE((unused)), int pe,
                                                         MPI_Win win)
{
    OSHMPI_CALLMPI(MPI_Win_flush_local(pe, win));
}

#include "mem_impl.h"
#include "coll_impl.h"
#include "rma_impl.h"
#include "amo_impl.h"
#include "order_impl.h"
#include "p2p_impl.h"
#include "lock_impl.h"
#include "am_progress_impl.h"

#endif /* OSHMPI_IMPL_H */

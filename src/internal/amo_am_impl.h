/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#ifndef INTERNAL_AMO_AM_IMPL_H
#define INTERNAL_AMO_AM_IMPL_H

#include "oshmpi_impl.h"

/* Ensure packet header variables can fit all possible types */
typedef union {
    int i_v;
    long l_v;
    long long ll_v;
    unsigned int ui_v;
    unsigned long ul_v;
    unsigned long long ull_v;
    int32_t i32_v;
    int64_t i64_v;
    uint32_t ui32_v;
    uint64_t ui64_v;
    size_t sz_v;
    ptrdiff_t ptr_v;
    float f_v;
    double d_v;
} OSHMPI_amo_datatype_t;

/* Define AMO active message packet header. Note that we do not define
 * packet for ACK of each packet, because all routines that require ACK
 * are blocking, thus we can directly receive ACK without going through
 * the callback progress handling. */
typedef enum {
    OSHMPI_AMO_PKT_CSWAP,
    OSHMPI_AMO_PKT_FETCH,
    OSHMPI_AMO_PKT_POST,
    OSHMPI_AMO_PKT_FLUSH,
    OSHMPI_AMO_PKT_TERMINATE,
    OSHMPI_AMO_PKT_MAX,
} OSHMPI_amo_pkt_type_t;

typedef struct OSHMPI_amo_cswap_pkt {
    OSHMPI_amo_datatype_t cond;
    OSHMPI_amo_datatype_t value;
    OSHMPI_symm_obj_type_t symm_obj_type;
    OSHMPI_amo_mpi_datatype_index_t mpi_type_idx;
    MPI_Aint target_disp;
    size_t bytes;
} OSHMPI_amo_cswap_pkt_t;

typedef struct OSHMPI_amo_fetch_pkt {
    OSHMPI_amo_datatype_t value;
    OSHMPI_symm_obj_type_t symm_obj_type;
    OSHMPI_amo_mpi_datatype_index_t mpi_type_idx;
    OSHMPI_amo_mpi_op_index_t mpi_op_idx;
    MPI_Aint target_disp;
    size_t bytes;
} OSHMPI_amo_fetch_pkt_t;

typedef OSHMPI_amo_fetch_pkt_t OSHMPI_amo_post_pkt_t;
typedef struct {
} OSHMPI_amo_flush_pkt_t;

typedef struct OSHMPI_amo_pkt {
    int type;
    union {
        OSHMPI_amo_cswap_pkt_t cswap;
        OSHMPI_amo_fetch_pkt_t fetch;
        OSHMPI_amo_post_pkt_t post;
        OSHMPI_amo_flush_pkt_t flush;
    };
} OSHMPI_amo_pkt_t;

#define OSHMPI_AMO_PKT_TAG 2000
#define OSHMPI_AMO_TERMINATE_TAG 2001
#define OSHMPI_AMO_PKT_ACK_TAG 2002

#define OSHMPI_AMO_OP_TYPE_IMPL(mpi_type_idx) do {              \
    switch(mpi_type_idx) {                                      \
        case OSHMPI_AMO_MPI_INT:                                \
           OSHMPI_OP_INT_MACRO(int); break;                     \
        case OSHMPI_AMO_MPI_LONG:                               \
           OSHMPI_OP_INT_MACRO(long); break;                    \
        case OSHMPI_AMO_MPI_LONG_LONG:                          \
           OSHMPI_OP_INT_MACRO(long long); break;               \
        case OSHMPI_AMO_MPI_UNSIGNED:                           \
           OSHMPI_OP_INT_MACRO(unsigned int); break;            \
        case OSHMPI_AMO_MPI_UNSIGNED_LONG:                      \
           OSHMPI_OP_INT_MACRO(unsigned long); break;           \
        case OSHMPI_AMO_MPI_UNSIGNED_LONG_LONG:                 \
           OSHMPI_OP_INT_MACRO(unsigned long long); break;      \
        case OSHMPI_AMO_MPI_INT32_T:                            \
           OSHMPI_OP_INT_MACRO(int32_t); break;                 \
        case OSHMPI_AMO_MPI_INT64_T:                            \
           OSHMPI_OP_INT_MACRO(int64_t); break;                 \
        case OSHMPI_AMO_OSHMPI_MPI_SIZE_T:                      \
           OSHMPI_OP_INT_MACRO(size_t); break;                  \
        case OSHMPI_AMO_MPI_UINT32_T:                           \
           OSHMPI_OP_INT_MACRO(uint32_t); break;                \
        case OSHMPI_AMO_MPI_UINT64_T:                           \
           OSHMPI_OP_INT_MACRO(uint64_t); break;                \
        case OSHMPI_AMO_OSHMPI_MPI_PTRDIFF_T:                   \
           OSHMPI_OP_INT_MACRO(ptrdiff_t); break;               \
        case OSHMPI_AMO_MPI_FLOAT:                              \
           OSHMPI_OP_FP_MACRO(float); break;                    \
        case OSHMPI_AMO_MPI_DOUBLE:                             \
           OSHMPI_OP_FP_MACRO(double); break;                   \
        default:                                                \
            OSHMPI_ERR_ABORT("Unsupported MPI type index: %d\n", mpi_type_idx);   \
            break;                                              \
    }                                                           \
} while (0)

#define OSHMPI_AMO_OP_FP_IMPL(mpi_op_idx, c_type, a_ptr, b_ptr)              \
        switch(mpi_op_idx) {                                                            \
            case OSHMPI_AMO_MPI_NO_OP:                                                  \
                break;                                                                  \
            case OSHMPI_AMO_MPI_REPLACE:                                                \
                *(c_type *) (b_ptr) = *(c_type *) (a_ptr);                              \
                break;                                                                  \
            case OSHMPI_AMO_MPI_SUM:                                                    \
                *(c_type *) (b_ptr) += *(c_type *) (a_ptr);                             \
                break;                                                                  \
            default:                                                                    \
                OSHMPI_ERR_ABORT("Unsupported MPI op index for floating point: %d\n", (int) mpi_op_idx);   \
                break;                                                                                     \
        }

#define OSHMPI_AMO_OP_INT_IMPL(mpi_op_idx, c_type, a_ptr, b_ptr)              \
        switch(mpi_op_idx) {                                                            \
            case OSHMPI_AMO_MPI_BAND:                                                   \
                *(c_type *) (b_ptr) = (*(c_type *) (a_ptr)) & (*(c_type *) (b_ptr));    \
                break;                                                                  \
            case OSHMPI_AMO_MPI_BOR:                                                    \
                *(c_type *) (b_ptr) = (*(c_type *) (a_ptr)) | (*(c_type *) (b_ptr));    \
                break;                                                                  \
            case OSHMPI_AMO_MPI_BXOR:                                                   \
                *(c_type *) (b_ptr) = (*(c_type *) (a_ptr)) ^ (*(c_type *) (b_ptr));    \
                break;                                                                  \
            case OSHMPI_AMO_MPI_NO_OP:                                                  \
                break;                                                                  \
            case OSHMPI_AMO_MPI_REPLACE:                                                \
                *(c_type *) (b_ptr) = *(c_type *) (a_ptr);                              \
                break;                                                                  \
            case OSHMPI_AMO_MPI_SUM:                                                    \
                *(c_type *) (b_ptr) += *(c_type *) (a_ptr);                             \
                break;                                                                  \
            default:                                                                    \
                OSHMPI_ERR_ABORT("Unsupported MPI op index for integer: %d\n", (int) mpi_op_idx);   \
                break;                                                                              \
        }

/* Callback of compare_and_swap AMO operation. */
OSHMPI_STATIC_INLINE_PREFIX void amo_cswap_pkt_cb(int origin_rank, OSHMPI_amo_pkt_t * pkt)
{
    OSHMPI_amo_datatype_t oldval;
    OSHMPI_amo_cswap_pkt_t *cswap_pkt = &pkt->cswap;
    void *dest = NULL;
    void *oldval_ptr = &oldval, *cond_ptr = &cswap_pkt->cond, *value_ptr = &cswap_pkt->value;

    OSHMPI_translate_disp_to_vaddr(cswap_pkt->symm_obj_type, cswap_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    /* Compute.
     * All AMOs are handled as active message, no lock needed.
     * We use same macro for floating point and integer types. */
#undef OSHMPI_OP_INT_MACRO
#undef OSHMPI_OP_FP_MACRO
#define OSHMPI_OP_INT_MACRO(c_type) do {           \
        *(c_type *) oldval_ptr = *(c_type *) dest;    \
        if (*(c_type *) dest == *(c_type *) cond_ptr) \
            *(c_type *) dest = *(c_type *) value_ptr; \
    } while (0)
#define OSHMPI_OP_FP_MACRO OSHMPI_OP_INT_MACRO
    OSHMPI_AMO_OP_TYPE_IMPL(cswap_pkt->mpi_type_idx);
#undef OSHMPI_OP_INT_MACRO
#undef OSHMPI_OP_FP_MACRO

    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Send(&oldval, 1, OSHMPI_global.amo_datatypes_table[cswap_pkt->mpi_type_idx],
                            origin_rank, OSHMPI_AMO_PKT_ACK_TAG, OSHMPI_global.amo_ack_comm_world));
}

/* Callback of fetch (with op) AMO operation. */
OSHMPI_STATIC_INLINE_PREFIX void amo_fetch_pkt_cb(int origin_rank, OSHMPI_amo_pkt_t * pkt)
{
    OSHMPI_amo_datatype_t oldval;
    OSHMPI_amo_fetch_pkt_t *fetch_pkt = &pkt->fetch;
    void *dest = NULL;
    void *oldval_ptr = &oldval, *value_ptr = &fetch_pkt->value;

    OSHMPI_translate_disp_to_vaddr(fetch_pkt->symm_obj_type, fetch_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    /* Compute.
     * All AMOs are handled as active message, no lock needed.
     * We use different op set for floating point and integer types. */
#undef OSHMPI_OP_INT_MACRO
#undef OSHMPI_OP_FP_MACRO
#define OSHMPI_OP_INT_MACRO(c_type) do {                          \
        *(c_type *) oldval_ptr = *(c_type *) dest;                \
        OSHMPI_AMO_OP_INT_IMPL(fetch_pkt->mpi_op_idx, c_type, value_ptr, dest); \
    } while (0)
#define OSHMPI_OP_FP_MACRO(c_type) do {                          \
        *(c_type *) oldval_ptr = *(c_type *) dest;               \
        OSHMPI_AMO_OP_FP_IMPL(fetch_pkt->mpi_op_idx, c_type, value_ptr, dest); \
    } while (0)

    OSHMPI_AMO_OP_TYPE_IMPL(fetch_pkt->mpi_type_idx);

#undef OSHMPI_OP_INT_MACRO
#undef OSHMPI_OP_FP_MACRO

    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Send(&oldval, 1, OSHMPI_global.amo_datatypes_table[fetch_pkt->mpi_type_idx],
                            origin_rank, OSHMPI_AMO_PKT_ACK_TAG, OSHMPI_global.amo_ack_comm_world));
}

/* Callback of post AMO operation. No ACK is returned to origin PE. */
OSHMPI_STATIC_INLINE_PREFIX void amo_post_pkt_cb(int origin_rank, OSHMPI_amo_pkt_t * pkt)
{
    void *dest = NULL;
    OSHMPI_amo_post_pkt_t *post_pkt = &pkt->post;
    void *value_ptr = &post_pkt->value;

    OSHMPI_translate_disp_to_vaddr(post_pkt->symm_obj_type, post_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    /* Compute.
     * All AMOs are handled as active message, no lock needed.
     * We use different op set for floating point and integer types. */
#undef OSHMPI_OP_INT_MACRO
#undef OSHMPI_OP_FP_MACRO

#define OSHMPI_OP_INT_MACRO(c_type) do {                          \
        OSHMPI_AMO_OP_INT_IMPL(post_pkt->mpi_op_idx, c_type, value_ptr, dest); \
    } while (0)
#define OSHMPI_OP_FP_MACRO(c_type) do {                          \
        OSHMPI_AMO_OP_FP_IMPL(post_pkt->mpi_op_idx, c_type, value_ptr, dest); \
    } while (0)

    OSHMPI_AMO_OP_TYPE_IMPL(post_pkt->mpi_type_idx);

#undef OSHMPI_OP_INT_MACRO
#undef OSHMPI_OP_FP_MACRO
}

/* Callback of flush synchronization. */
OSHMPI_STATIC_INLINE_PREFIX void amo_flush_pkt_cb(int origin_rank, OSHMPI_amo_pkt_t * pkt)
{
    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Send(NULL, 0, MPI_BYTE, origin_rank, OSHMPI_AMO_PKT_ACK_TAG,
                            OSHMPI_global.amo_ack_comm_world));
}

#if defined(OSHMPI_ENABLE_AMO_ASYNC_THREAD) || defined(OSHMPI_RUNTIME_AMO_ASYNC_THREAD)
/* Async thread blocking progress polling for AMO active message */
OSHMPI_STATIC_INLINE_PREFIX void *amo_cb_async_progress(void *arg OSHMPI_ATTRIBUTE((unused)))
{
    int cb_flag = 0;
    MPI_Status cb_stat;
    OSHMPI_amo_pkt_t *amo_pkt = (OSHMPI_amo_pkt_t *) OSHMPI_global.amo_pkt;

    /* Use amo_comm_world to send/receive AMO packets from remote PEs
     * or termination flag from the main thread; use amo_ack_comm_world
     * for ack send in callback in order to avoid interaction with AMO packets.
     * Fetch or flush issuing routine can directly receive the ack
     * without going through the callback progress. */
    while (1) {
        OSHMPI_CALLMPI(MPI_Test(&OSHMPI_global.amo_req, &cb_flag, &cb_stat));
        if (cb_flag) {
            OSHMPI_DBGMSG("received AMO packet origin %d, type %d\n",
                          cb_stat.MPI_SOURCE, amo_pkt->type);
            /* Handle packet */
            switch (amo_pkt->type) {
                case OSHMPI_AMO_PKT_CSWAP:
                    amo_cswap_pkt_cb(cb_stat.MPI_SOURCE, amo_pkt);
                    break;
                case OSHMPI_AMO_PKT_FETCH:
                    amo_fetch_pkt_cb(cb_stat.MPI_SOURCE, amo_pkt);
                    break;
                case OSHMPI_AMO_PKT_POST:
                    amo_post_pkt_cb(cb_stat.MPI_SOURCE, amo_pkt);
                    break;
                case OSHMPI_AMO_PKT_FLUSH:
                    amo_flush_pkt_cb(cb_stat.MPI_SOURCE, amo_pkt);
                    break;
                case OSHMPI_AMO_PKT_TERMINATE:
                    OSHMPI_DBGMSG("received terminate\n");
                    goto terminate;     /* The main thread reached finalize */
                    break;
                default:
                    OSHMPI_ERR_ABORT("Unsupported AMO packet type: %d\n", amo_pkt->type);
                    break;
            }

            /* Post next receive */
            OSHMPI_CALLMPI(MPI_Irecv(amo_pkt, sizeof(OSHMPI_amo_pkt_t),
                                     MPI_BYTE, MPI_ANY_SOURCE,
                                     MPI_ANY_TAG, OSHMPI_global.amo_comm_world,
                                     &OSHMPI_global.amo_req));
        }
        /* THREAD_YIELD */
    }

  terminate:
    OSHMPI_CALLPTHREAD(pthread_mutex_lock(&OSHMPI_global.amo_async_mutex));
    OSHMPI_global.amo_async_thread_done = 1;
    OSHMPI_CALLPTHREAD(pthread_mutex_unlock(&OSHMPI_global.amo_async_mutex));
    OSHMPI_CALLPTHREAD(pthread_cond_signal(&OSHMPI_global.amo_async_cond));

    return NULL;
}
#endif /* defined(OSHMPI_ENABLE_AMO_ASYNC_THREAD) || defined(OSHMPI_RUNTIME_AMO_ASYNC_THREAD) */

#if !defined(OSHMPI_ENABLE_AMO_ASYNC_THREAD)    /* disabled or runtime */
#define OSHMPI_AMO_AM_CB_PROGRESS_POLL_NCNT 1

/* Nonblocking polling progress for AMO active message. Triggered by each PE. */
OSHMPI_STATIC_INLINE_PREFIX void amo_am_cb_progress(void)
{
    int poll_cnt = OSHMPI_AMO_AM_CB_PROGRESS_POLL_NCNT, cb_flag = 0;
    MPI_Status cb_stat;
    OSHMPI_amo_pkt_t *amo_pkt = (OSHMPI_amo_pkt_t *) OSHMPI_global.amo_pkt;

    /* Only one thread can poll progress at a time */
    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.amo_cb_progress_cs);

    /* Use amo_comm_world to send/receive AMO packets from remote PEs
     * or termination flag from the main thread; use amo_ack_comm_world
     * for ack send in callback in order to avoid interaction with AMO packets.
     * Fetch or flush issuing routine can directly receive the ack
     * without going through the callback progress. */
    while (poll_cnt-- > 0) {
        OSHMPI_CALLMPI(MPI_Test(&OSHMPI_global.amo_req, &cb_flag, &cb_stat));
        if (cb_flag) {
            OSHMPI_DBGMSG("received AMO packet origin %d, type %d\n",
                          cb_stat.MPI_SOURCE, amo_pkt->type);
            /* Handle packet */
            switch (amo_pkt->type) {
                case OSHMPI_AMO_PKT_CSWAP:
                    amo_cswap_pkt_cb(cb_stat.MPI_SOURCE, amo_pkt);
                    break;
                case OSHMPI_AMO_PKT_FETCH:
                    amo_fetch_pkt_cb(cb_stat.MPI_SOURCE, amo_pkt);
                    break;
                case OSHMPI_AMO_PKT_POST:
                    amo_post_pkt_cb(cb_stat.MPI_SOURCE, amo_pkt);
                    break;
                case OSHMPI_AMO_PKT_FLUSH:
                    amo_flush_pkt_cb(cb_stat.MPI_SOURCE, amo_pkt);
                    break;
                case OSHMPI_AMO_PKT_TERMINATE:
                    OSHMPI_DBGMSG("received terminate\n");
                    break;      /* Reached finalize */
                default:
                    OSHMPI_ERR_ABORT("Unsupported AMO packet type: %d\n",
                                     OSHMPI_global.amo_pkt->type);
                    break;
            }

            /* Post next receive */
            OSHMPI_CALLMPI(MPI_Irecv(amo_pkt, sizeof(OSHMPI_amo_pkt_t),
                                     MPI_BYTE, MPI_ANY_SOURCE,
                                     OSHMPI_AMO_PKT_TAG, OSHMPI_global.amo_comm_world,
                                     &OSHMPI_global.amo_req));
        }
    }

    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.amo_cb_progress_cs);
}
#endif /* OSHMPI_ENABLE_AMO_ASYNC_THREAD */

#if defined(OSHMPI_ENABLE_AMO_ASYNC_THREAD)
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_cb_progress(void)
{

}
#elif defined(OSHMPI_RUNTIME_AMO_ASYNC_THREAD)
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_cb_progress(void)
{
    /* Make progress only when async thread is disabled */
    if (!OSHMPI_env.enable_async_thread)
        amo_am_cb_progress();
}
#else /* enable at configure */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_cb_progress(void)
{
    amo_am_cb_progress();
}
#endif

OSHMPI_STATIC_INLINE_PREFIX void amo_cb_progress_start(void)
{
    OSHMPI_amo_pkt_t *amo_pkt = (OSHMPI_amo_pkt_t *) OSHMPI_global.amo_pkt;

    /* Post first receive */
    OSHMPI_CALLMPI(MPI_Irecv(amo_pkt, sizeof(OSHMPI_amo_pkt_t),
                             MPI_BYTE, MPI_ANY_SOURCE,
                             OSHMPI_AMO_PKT_TAG, OSHMPI_global.amo_comm_world,
                             &OSHMPI_global.amo_req));

#if defined(OSHMPI_ENABLE_AMO_ASYNC_THREAD) || defined(OSHMPI_RUNTIME_AMO_ASYNC_THREAD)
    if (OSHMPI_env.enable_async_thread) {
        pthread_attr_t attr;

        OSHMPI_global.amo_async_thread_done = 0;
        OSHMPI_CALLPTHREAD(pthread_mutex_init(&OSHMPI_global.amo_async_mutex, NULL));
        OSHMPI_CALLPTHREAD(pthread_cond_init(&OSHMPI_global.amo_async_cond, NULL));

        /* Create asynchronous progress thread */
        OSHMPI_CALLPTHREAD(pthread_attr_init(&attr));
        OSHMPI_CALLPTHREAD(pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED));
        OSHMPI_CALLPTHREAD(pthread_create
                           (&OSHMPI_global.amo_async_thread, &attr, &amo_cb_async_progress, NULL));
        OSHMPI_CALLPTHREAD(pthread_attr_destroy(&attr));
    }
#endif
}

OSHMPI_STATIC_INLINE_PREFIX void amo_cb_progress_end(void)
{
    MPI_Status terminate_stat OSHMPI_ATTRIBUTE((unused));
    OSHMPI_amo_pkt_t pkt;
    pkt.type = OSHMPI_AMO_PKT_TERMINATE;

    /* Sent terminate signal to progress polling thread or consume the
     * last outstanding irecv on calling PE. */
    MPI_Send(&pkt, sizeof(OSHMPI_amo_pkt_t), MPI_BYTE, OSHMPI_global.world_rank,
             OSHMPI_AMO_PKT_TAG, OSHMPI_global.amo_comm_world);
    OSHMPI_DBGMSG("sent terminate\n");

#if defined(OSHMPI_ENABLE_AMO_ASYNC_THREAD) || defined(OSHMPI_RUNTIME_AMO_ASYNC_THREAD)
    if (OSHMPI_env.enable_async_thread) {
        OSHMPI_CALLPTHREAD(pthread_mutex_lock(&OSHMPI_global.amo_async_mutex));
        while (!OSHMPI_global.amo_async_thread_done) {
            OSHMPI_CALLPTHREAD(pthread_cond_wait
                               (&OSHMPI_global.amo_async_cond, &OSHMPI_global.amo_async_mutex));
        }
        OSHMPI_CALLPTHREAD(pthread_mutex_unlock(&OSHMPI_global.amo_async_mutex));
        OSHMPI_CALLPTHREAD(pthread_cond_destroy(&OSHMPI_global.amo_async_cond));
        OSHMPI_CALLPTHREAD(pthread_mutex_destroy(&OSHMPI_global.amo_async_mutex));
        return; /* thread already completed last irecv */
    }
#endif

    /* Complete last irecv on calling PE. */
    OSHMPI_CALLMPI(MPI_Wait(&OSHMPI_global.amo_req, &terminate_stat));
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_initialize(void)
{
    /* Dup comm world for the AMO progress thread */
    OSHMPI_CALLMPI(MPI_Comm_dup(OSHMPI_global.comm_world, &OSHMPI_global.amo_comm_world));
    OSHMPI_CALLMPI(MPI_Comm_dup(OSHMPI_global.comm_world, &OSHMPI_global.amo_ack_comm_world));

    /* Per PE flag indicating outstanding AM AMOs. */
    OSHMPI_global.amo_outstanding_op_flags =
        OSHMPIU_malloc(sizeof(OSHMPI_atomic_flag_t) * OSHMPI_global.world_size);
    OSHMPI_ASSERT(OSHMPI_global.amo_outstanding_op_flags);
    memset(OSHMPI_global.amo_outstanding_op_flags, 0,
           sizeof(OSHMPI_atomic_flag_t) * OSHMPI_global.world_size);

    /* Global datatype table used for index translation */
    OSHMPI_global.amo_datatypes_table =
        OSHMPIU_malloc(sizeof(MPI_Datatype) * OSHMPI_AMO_MPI_DATATYPE_MAX);
    OSHMPI_ASSERT(OSHMPI_global.amo_datatypes_table);
    OSHMPI_global.amo_datatypes_table[OSHMPI_AMO_MPI_INT] = MPI_INT;
    OSHMPI_global.amo_datatypes_table[OSHMPI_AMO_MPI_LONG] = MPI_LONG;
    OSHMPI_global.amo_datatypes_table[OSHMPI_AMO_MPI_LONG_LONG] = MPI_LONG_LONG;
    OSHMPI_global.amo_datatypes_table[OSHMPI_AMO_MPI_UNSIGNED] = MPI_UNSIGNED;
    OSHMPI_global.amo_datatypes_table[OSHMPI_AMO_MPI_UNSIGNED_LONG] = MPI_UNSIGNED_LONG;
    OSHMPI_global.amo_datatypes_table[OSHMPI_AMO_MPI_UNSIGNED_LONG_LONG] = MPI_UNSIGNED_LONG_LONG;
    OSHMPI_global.amo_datatypes_table[OSHMPI_AMO_MPI_INT32_T] = MPI_INT32_T;
    OSHMPI_global.amo_datatypes_table[OSHMPI_AMO_MPI_INT64_T] = MPI_INT64_T;
    OSHMPI_global.amo_datatypes_table[OSHMPI_AMO_MPI_UINT32_T] = MPI_UINT32_T;
    OSHMPI_global.amo_datatypes_table[OSHMPI_AMO_MPI_UINT64_T] = MPI_UINT64_T;
    OSHMPI_global.amo_datatypes_table[OSHMPI_AMO_OSHMPI_MPI_SIZE_T] = OSHMPI_MPI_SIZE_T;
    OSHMPI_global.amo_datatypes_table[OSHMPI_AMO_OSHMPI_MPI_PTRDIFF_T] = OSHMPI_MPI_PTRDIFF_T;
    OSHMPI_global.amo_datatypes_table[OSHMPI_AMO_MPI_FLOAT] = MPI_FLOAT;
    OSHMPI_global.amo_datatypes_table[OSHMPI_AMO_MPI_DOUBLE] = MPI_DOUBLE;

    /* Global op table used for index translation */
    OSHMPI_global.amo_ops_table = OSHMPIU_malloc(sizeof(MPI_Op) * OSHMPI_AMO_MPI_OP_MAX);
    OSHMPI_ASSERT(OSHMPI_global.amo_ops_table);
    OSHMPI_global.amo_ops_table[OSHMPI_AMO_MPI_BAND] = MPI_BAND;
    OSHMPI_global.amo_ops_table[OSHMPI_AMO_MPI_BOR] = MPI_BOR;
    OSHMPI_global.amo_ops_table[OSHMPI_AMO_MPI_BXOR] = MPI_BXOR;
    OSHMPI_global.amo_ops_table[OSHMPI_AMO_MPI_NO_OP] = MPI_NO_OP;
    OSHMPI_global.amo_ops_table[OSHMPI_AMO_MPI_REPLACE] = MPI_REPLACE;
    OSHMPI_global.amo_ops_table[OSHMPI_AMO_MPI_SUM] = MPI_SUM;

    OSHMPI_global.amo_pkt = OSHMPIU_malloc(sizeof(OSHMPI_amo_pkt_t));
    OSHMPI_ASSERT(OSHMPI_global.amo_pkt);

    OSHMPI_THREAD_INIT_CS(&OSHMPI_global.amo_cb_progress_cs);
    amo_cb_progress_start();

    OSHMPI_DBGMSG("Initialized active message AMO\n");
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_finalize(void)
{

    /* The finalize routine has to be called after implicity barrier
     * in shmem_finalize to ensure no incoming AMO request */

    amo_cb_progress_end();
    OSHMPI_THREAD_DESTROY_CS(&OSHMPI_global.amo_cb_progress_cs);

    OSHMPI_CALLMPI(MPI_Comm_free(&OSHMPI_global.amo_comm_world));
    OSHMPI_CALLMPI(MPI_Comm_free(&OSHMPI_global.amo_ack_comm_world));
    OSHMPIU_free(OSHMPI_global.amo_outstanding_op_flags);
    OSHMPIU_free(OSHMPI_global.amo_datatypes_table);
    OSHMPIU_free(OSHMPI_global.amo_ops_table);
    OSHMPIU_free(OSHMPI_global.amo_pkt);
}

/* Issue a compare_and_swap operation. Blocking wait until return of old value. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_cswap(shmem_ctx_t ctx
                                                     OSHMPI_ATTRIBUTE((unused)),
                                                     MPI_Datatype mpi_type,
                                                     OSHMPI_amo_mpi_datatype_index_t mpi_type_idx,
                                                     size_t bytes, void *dest, void *cond_ptr,
                                                     void *value_ptr, int pe, void *oldval_ptr)
{
    OSHMPI_amo_pkt_t pkt;
    OSHMPI_amo_cswap_pkt_t *cswap_pkt = &pkt.cswap;
    MPI_Aint target_disp = -1;
    MPI_Win win = MPI_WIN_NULL;

    OSHMPI_translate_win_and_disp((const void *) dest, &win, &target_disp);
    OSHMPI_ASSERT(target_disp >= 0 && win != MPI_WIN_NULL);

    pkt.type = OSHMPI_AMO_PKT_CSWAP;
    memcpy(&cswap_pkt->cond, cond_ptr, bytes);
    memcpy(&cswap_pkt->value, value_ptr, bytes);
    cswap_pkt->target_disp = target_disp;
    cswap_pkt->mpi_type_idx = mpi_type_idx;
    cswap_pkt->bytes = bytes;
    cswap_pkt->symm_obj_type = (win == OSHMPI_global.symm_heap_win) ?
        OSHMPI_SYMM_OBJ_HEAP : OSHMPI_SYMM_OBJ_DATA;

    OSHMPI_am_progress_mpi_send(&pkt, sizeof(OSHMPI_amo_pkt_t), MPI_BYTE, pe, OSHMPI_AMO_PKT_TAG,
                                OSHMPI_global.amo_comm_world);

    OSHMPI_am_progress_mpi_recv(oldval_ptr, 1, mpi_type, pe, OSHMPI_AMO_PKT_ACK_TAG,
                                OSHMPI_global.amo_ack_comm_world, MPI_STATUS_IGNORE);

    OSHMPI_DBGMSG("packet type %d, symm type %s, target %d, datatype idx %d\n",
                  pkt.type, (cswap_pkt->symm_obj_type == OSHMPI_SYMM_OBJ_HEAP) ? "heap" : "data",
                  pe, mpi_type_idx);

    /* Reset flag since remote PE should have finished previous post
     * before handling this fetch. */
    OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_global.amo_outstanding_op_flags[pe], 0);
}

/* Issue a fetch (with op) operation. Blocking wait until return of old value. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_fetch(shmem_ctx_t ctx
                                                     OSHMPI_ATTRIBUTE((unused)),
                                                     MPI_Datatype mpi_type,
                                                     OSHMPI_amo_mpi_datatype_index_t mpi_type_idx,
                                                     size_t bytes, MPI_Op op,
                                                     OSHMPI_amo_mpi_op_index_t op_idx, void *dest,
                                                     void *value_ptr, int pe, void *oldval_ptr)
{
    OSHMPI_amo_pkt_t pkt;
    OSHMPI_amo_fetch_pkt_t *fetch_pkt = &pkt.fetch;
    MPI_Aint target_disp = -1;
    MPI_Win win = MPI_WIN_NULL;

    OSHMPI_translate_win_and_disp((const void *) dest, &win, &target_disp);
    OSHMPI_ASSERT(target_disp >= 0 && win != MPI_WIN_NULL);

    pkt.type = OSHMPI_AMO_PKT_FETCH;
    fetch_pkt->target_disp = target_disp;
    fetch_pkt->mpi_type_idx = mpi_type_idx;
    fetch_pkt->mpi_op_idx = op_idx;
    fetch_pkt->bytes = bytes;
    if (fetch_pkt->mpi_op_idx != OSHMPI_AMO_MPI_NO_OP)
        memcpy(&fetch_pkt->value, value_ptr, bytes);    /* ignore value in atomic-fetch */
    fetch_pkt->symm_obj_type = (win == OSHMPI_global.symm_heap_win) ?
        OSHMPI_SYMM_OBJ_HEAP : OSHMPI_SYMM_OBJ_DATA;

    OSHMPI_am_progress_mpi_send(&pkt, sizeof(OSHMPI_amo_pkt_t), MPI_BYTE, pe, OSHMPI_AMO_PKT_TAG,
                                OSHMPI_global.amo_comm_world);

    OSHMPI_am_progress_mpi_recv(oldval_ptr, 1, mpi_type, pe, OSHMPI_AMO_PKT_ACK_TAG,
                                OSHMPI_global.amo_ack_comm_world, MPI_STATUS_IGNORE);

    OSHMPI_DBGMSG("packet type %d, symm type %s, target %d, datatype idx %d, op idx %d\n",
                  pkt.type, (fetch_pkt->symm_obj_type == OSHMPI_SYMM_OBJ_HEAP) ? "heap" : "data",
                  pe, mpi_type_idx, op_idx);

    /* Reset flag since remote PE should have finished previous post
     * before handling this fetch. */
    OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_global.amo_outstanding_op_flags[pe], 0);
}

/* Issue a post operation. Return immediately after sent AMO packet */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_post(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                    MPI_Datatype mpi_type,
                                                    OSHMPI_amo_mpi_datatype_index_t mpi_type_idx,
                                                    size_t bytes, MPI_Op op,
                                                    OSHMPI_amo_mpi_op_index_t op_idx, void *dest,
                                                    void *value_ptr, int pe)
{
    OSHMPI_amo_pkt_t pkt;
    OSHMPI_amo_post_pkt_t *post_pkt = &pkt.post;
    MPI_Aint target_disp = -1;
    MPI_Win win = MPI_WIN_NULL;

    OSHMPI_translate_win_and_disp((const void *) dest, &win, &target_disp);
    OSHMPI_ASSERT(target_disp >= 0 && win != MPI_WIN_NULL);

    pkt.type = OSHMPI_AMO_PKT_POST;
    post_pkt->target_disp = target_disp;
    post_pkt->mpi_type_idx = mpi_type_idx;
    post_pkt->mpi_op_idx = op_idx;
    post_pkt->bytes = bytes;
    memcpy(&post_pkt->value, value_ptr, bytes);
    post_pkt->symm_obj_type = (win == OSHMPI_global.symm_heap_win) ?
        OSHMPI_SYMM_OBJ_HEAP : OSHMPI_SYMM_OBJ_DATA;

    OSHMPI_am_progress_mpi_send(&pkt, sizeof(OSHMPI_amo_pkt_t), MPI_BYTE, pe, OSHMPI_AMO_PKT_TAG,
                                OSHMPI_global.amo_comm_world);
    OSHMPI_DBGMSG("packet type %d, symm type %s, target %d, datatype idx %d, op idx %d\n",
                  pkt.type, (post_pkt->symm_obj_type == OSHMPI_SYMM_OBJ_HEAP) ? "heap" : "data",
                  pe, mpi_type_idx, op_idx);

    /* Indicate outstanding AMO */
    OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_global.amo_outstanding_op_flags[pe], 1);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_flush(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                     int PE_start, int logPE_stride, int PE_size)
{
    OSHMPI_amo_pkt_t pkt;
    int pe, nreqs = 0, noutstanding_pes = 0, i;
    MPI_Request *reqs = NULL;
    const int pe_stride = 1 << logPE_stride;    /* Implement 2^pe_logs with bitshift. */

    for (i = 0; i < PE_size; i++) {
        pe = PE_start + i * pe_stride;
        if (OSHMPI_ATOMIC_FLAG_LOAD(OSHMPI_global.amo_outstanding_op_flags[pe]))
            noutstanding_pes++;
    }

    /* Do nothing if no PE has outstanding AMOs */
    if (noutstanding_pes == 0) {
        OSHMPI_DBGMSG("skipped all [start %d, stride %d, size %d]\n",
                      PE_start, logPE_stride, PE_size);
        return;
    }

    /* Issue a flush synchronization to remote PEs.
     * Threaded: the flag might be concurrently updated by another thread,
     * thus we always allocate reqs for all PEs in the active set.*/
    reqs = OSHMPIU_malloc(sizeof(MPI_Request) * PE_size * 2);
    OSHMPI_ASSERT(reqs);
    pkt.type = OSHMPI_AMO_PKT_FLUSH;

    for (i = 0; i < PE_size; i++) {
        pe = PE_start + i * pe_stride;
        if (OSHMPI_ATOMIC_FLAG_LOAD(OSHMPI_global.amo_outstanding_op_flags[pe])) {
            OSHMPI_CALLMPI(MPI_Isend
                           (&pkt, sizeof(OSHMPI_amo_pkt_t), MPI_BYTE, pe, OSHMPI_AMO_PKT_TAG,
                            OSHMPI_global.amo_comm_world, &reqs[nreqs++]));
            OSHMPI_CALLMPI(MPI_Irecv
                           (NULL, 0, MPI_BYTE, pe, OSHMPI_AMO_PKT_ACK_TAG,
                            OSHMPI_global.amo_ack_comm_world, &reqs[nreqs++]));
            OSHMPI_DBGMSG("packet type %d, target %d in [start %d, stride %d, size %d]\n",
                          pkt.type, pe, PE_start, logPE_stride, PE_size);
        }
    }

    OSHMPI_ASSERT(PE_size * 2 >= nreqs);
    OSHMPI_am_progress_mpi_waitall(nreqs, reqs, MPI_STATUS_IGNORE);
    OSHMPIU_free(reqs);

    /* Reset all flags
     * Threaded: the flag might be concurrently updated by another thread,
     * however, the user must use additional thread sync to ensure a flush
     * completes the specific AMO. */
    for (i = 0; i < PE_size; i++) {
        pe = PE_start + i * pe_stride;
        OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_global.amo_outstanding_op_flags[pe], 0);
    }
}

/* Issue a flush synchronization to ensure completion of all outstanding AMOs to remote PEs.
 * Blocking wait until received ACK from remote PE. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_flush_all(shmem_ctx_t ctx)
{
    OSHMPI_amo_am_flush(ctx, 0 /* PE_start */ , 0 /* logPE_stride */ ,
                        OSHMPI_global.world_size /* PE_size */);
}

#ifdef OSHMPI_ENABLE_AM_AMO
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_initialize(void)
{
    OSHMPI_amo_am_initialize();
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_finalize(void)
{
    OSHMPI_amo_am_finalize();
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_cb_progress(void)
{
    OSHMPI_amo_am_cb_progress();
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_cswap(shmem_ctx_t ctx
                                                  OSHMPI_ATTRIBUTE((unused)), MPI_Datatype mpi_type,
                                                  OSHMPI_amo_mpi_datatype_index_t mpi_type_idx,
                                                  size_t bytes, void *dest /* target_addr */ ,
                                                  void *cond_ptr /*compare_addr */ ,
                                                  void *value_ptr /* origin_addr */ ,
                                                  int pe, void *oldval_ptr /*result_addr */)
{
    OSHMPI_amo_am_cswap(ctx, mpi_type, mpi_type_idx, bytes, dest, cond_ptr,
                        value_ptr, pe, oldval_ptr);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_fetch(shmem_ctx_t ctx
                                                  OSHMPI_ATTRIBUTE((unused)), MPI_Datatype mpi_type,
                                                  OSHMPI_amo_mpi_datatype_index_t mpi_type_idx,
                                                  size_t bytes, MPI_Op op,
                                                  OSHMPI_amo_mpi_op_index_t op_idx,
                                                  void *dest /* target_addr */ ,
                                                  void *value_ptr /* origin_addr */ ,
                                                  int pe, void *oldval_ptr /* result_addr */)
{
    OSHMPI_amo_am_fetch(ctx, mpi_type, mpi_type_idx, bytes, op, op_idx, dest,
                        value_ptr, pe, oldval_ptr);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_post(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 MPI_Datatype mpi_type,
                                                 OSHMPI_amo_mpi_datatype_index_t mpi_type_idx,
                                                 size_t bytes, MPI_Op op,
                                                 OSHMPI_amo_mpi_op_index_t op_idx,
                                                 void *dest /* target_addr */ ,
                                                 void *value_ptr /* origin_addr */ ,
                                                 int pe)
{
    OSHMPI_amo_am_post(ctx, mpi_type, mpi_type_idx, bytes, op, op_idx, dest, value_ptr, pe);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_flush(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                  int PE_start, int logPE_stride, int PE_size)
{
    OSHMPI_amo_am_flush(ctx, PE_start, logPE_stride, PE_size);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_flush_all(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)))
{
    OSHMPI_amo_am_flush_all(ctx);
}
#endif /* OSHMPI_ENABLE_AM_AMO */
#endif /* INTERNAL_AMO_AM_IMPL_H */

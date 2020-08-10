/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_AM_IMPL_H
#define INTERNAL_AM_IMPL_H

#include "oshmpi_impl.h"
#include "amo_am_impl.h"
#include "rma_am_impl.h"

/* Callback of flush synchronization. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_flush_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt)
{
    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Send(NULL, 0, MPI_BYTE, origin_rank, OSHMPI_AM_PKT_ACK_TAG,
                            OSHMPI_global.am_ack_comm_world));
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_flush(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 int PE_start, int logPE_stride, int PE_size)
{
    OSHMPI_am_pkt_t pkt;
    int pe, nreqs = 0, noutstanding_pes = 0, i;
    MPI_Request *reqs = NULL;
    const int pe_stride = 1 << logPE_stride;    /* Implement 2^pe_logs with bitshift. */

    /* No AM flush is needed if direct AMO is enabled. */
    if (OSHMPI_ENABLE_DIRECT_AMO_RUNTIME)
        return;

    for (i = 0; i < PE_size; i++) {
        pe = PE_start + i * pe_stride;
        if (OSHMPI_ATOMIC_FLAG_LOAD(OSHMPI_global.am_outstanding_op_flags[pe]))
            noutstanding_pes++;
    }

    /* Do nothing if no PE has outstanding AMOs */
    if (noutstanding_pes == 0)
        return;

    /* Issue a flush synchronization to remote PEs.
     * Threaded: the flag might be concurrently updated by another thread,
     * thus we always allocate reqs for all PEs in the active set.*/
    reqs = OSHMPIU_malloc(sizeof(MPI_Request) * PE_size * 2);
    OSHMPI_ASSERT(reqs);
    pkt.type = OSHMPI_AM_PKT_FLUSH;

    for (i = 0; i < PE_size; i++) {
        pe = PE_start + i * pe_stride;
        if (OSHMPI_ATOMIC_FLAG_LOAD(OSHMPI_global.am_outstanding_op_flags[pe])) {
            OSHMPI_CALLMPI(MPI_Isend
                           (&pkt, sizeof(OSHMPI_am_pkt_t), MPI_BYTE, pe, OSHMPI_AM_PKT_TAG,
                            OSHMPI_global.am_comm_world, &reqs[nreqs++]));
            OSHMPI_CALLMPI(MPI_Irecv
                           (NULL, 0, MPI_BYTE, pe, OSHMPI_AM_PKT_ACK_TAG,
                            OSHMPI_global.am_ack_comm_world, &reqs[nreqs++]));
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
        OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_global.am_outstanding_op_flags[pe], 0);
    }
}

/* Issue a flush synchronization to ensure completion of all outstanding AMOs to remote PEs.
 * Blocking wait until received ACK from remote PE. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_flush_all(shmem_ctx_t ctx)
{
    /* No AM flush is needed if direct AMO is enabled. */
    if (OSHMPI_ENABLE_DIRECT_AMO_RUNTIME)
        return;

    OSHMPI_am_flush(ctx, 0 /* PE_start */ , 0 /* logPE_stride */ ,
                    OSHMPI_global.world_size /* PE_size */);
}

OSHMPI_STATIC_INLINE_PREFIX void am_cb_handle_pkt(OSHMPI_am_pkt_t * am_pkt, int source_rank,
                                                  int *terminate_flag)
{
    *terminate_flag = 0;

    /* Handle packet.
     * Select from list of cb functions to avoid access overhead to function pointers */
    switch (am_pkt->type) {
        case OSHMPI_AM_PKT_CSWAP:
            OSHMPI_amo_am_cswap_pkt_cb(source_rank, am_pkt);
            break;
        case OSHMPI_AM_PKT_FETCH:
            OSHMPI_amo_am_fetch_pkt_cb(source_rank, am_pkt);
            break;
        case OSHMPI_AM_PKT_POST:
            OSHMPI_amo_am_post_pkt_cb(source_rank, am_pkt);
            break;
        case OSHMPI_AM_PKT_PUT:
            OSHMPI_rma_am_put_pkt_cb(source_rank, am_pkt);
            break;
        case OSHMPI_AM_PKT_GET:
            OSHMPI_rma_am_get_pkt_cb(source_rank, am_pkt);
            break;
        case OSHMPI_AM_PKT_IPUT:
            OSHMPI_rma_am_iput_pkt_cb(source_rank, am_pkt);
            break;
        case OSHMPI_AM_PKT_IGET:
            OSHMPI_rma_am_iget_pkt_cb(source_rank, am_pkt);
            break;
        case OSHMPI_AM_PKT_FLUSH:
            OSHMPI_am_flush_pkt_cb(source_rank, am_pkt);
            break;
        case OSHMPI_AM_PKT_TERMINATE:
            OSHMPI_DBGMSG("received terminate\n");
            *terminate_flag = 1;
            break;      /* Reached finalize */
        default:
            OSHMPI_ERR_ABORT("Unsupported AMO packet type: %d\n", am_pkt->type);
            break;
    }
}

/* Async thread blocking progress polling for AMO active message */
#if defined(OSHMPI_ENABLE_AM_ASYNC_THREAD) || defined(OSHMPI_RUNTIME_AM_ASYNC_THREAD)
OSHMPI_STATIC_INLINE_PREFIX void *am_cb_async_progress(void *arg OSHMPI_ATTRIBUTE((unused)))
{
    int cb_flag = 0;
    MPI_Status cb_stat;
    OSHMPI_am_pkt_t *am_pkt = (OSHMPI_am_pkt_t *) OSHMPI_global.am_pkt;

    /* Use am_comm_world to send/receive AMO packets from remote PEs
     * or termination flag from the main thread; use am_ack_comm_world
     * for ack send in callback in order to avoid interaction with AMO packets.
     * Fetch or flush issuing routine can directly receive the ack
     * without going through the callback progress. */
    while (1) {
        OSHMPI_CALLMPI(MPI_Test(&OSHMPI_global.am_req, &cb_flag, &cb_stat));
        if (cb_flag) {
            OSHMPI_DBGMSG("received AMO packet origin %d, type %d\n",
                          cb_stat.MPI_SOURCE, am_pkt->type);

            /* Handle packet */
            int terminate_flag = 0;
            am_cb_handle_pkt(am_pkt, cb_stat.MPI_SOURCE, &terminate_flag);
            if (terminate_flag)
                goto terminate; /* The main thread reached finalize */

            /* Post next receive */
            OSHMPI_CALLMPI(MPI_Irecv(am_pkt, sizeof(OSHMPI_am_pkt_t),
                                     MPI_BYTE, MPI_ANY_SOURCE,
                                     MPI_ANY_TAG, OSHMPI_global.am_comm_world,
                                     &OSHMPI_global.am_req));
        }
        /* THREAD_YIELD */
    }

  terminate:
    OSHMPI_CALLPTHREAD(pthread_mutex_lock(&OSHMPI_global.am_async_mutex));
    OSHMPI_global.am_async_thread_done = 1;
    OSHMPI_CALLPTHREAD(pthread_mutex_unlock(&OSHMPI_global.am_async_mutex));
    OSHMPI_CALLPTHREAD(pthread_cond_signal(&OSHMPI_global.am_async_cond));

    return NULL;
}
#else
OSHMPI_STATIC_INLINE_PREFIX void *am_cb_async_progress(void *arg OSHMPI_ATTRIBUTE((unused)))
{
    return NULL;
}
#endif /* end of defined(OSHMPI_ENABLE_AM_ASYNC_THREAD) || defined(OSHMPI_RUNTIME_AM_ASYNC_THREAD) */

#define OSHMPI_AM_CB_PROGRESS_POLL_NCNT 1

/* Nonblocking polling progress for AMO active message. Triggered by each PE. */
OSHMPI_STATIC_INLINE_PREFIX void am_cb_manual_progress(void)
{
    int poll_cnt = OSHMPI_AM_CB_PROGRESS_POLL_NCNT, cb_flag = 0;
    MPI_Status cb_stat;
    OSHMPI_am_pkt_t *am_pkt = (OSHMPI_am_pkt_t *) OSHMPI_global.am_pkt;

    /* Only one thread can poll progress at a time */
    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.am_cb_progress_cs);

    /* Use am_comm_world to send/receive AMO packets from remote PEs
     * or termination flag from the main thread; use am_ack_comm_world
     * for ack send in callback in order to avoid interaction with AMO packets.
     * Fetch or flush issuing routine can directly receive the ack
     * without going through the callback progress. */
    while (poll_cnt-- > 0) {
        OSHMPI_CALLMPI(MPI_Test(&OSHMPI_global.am_req, &cb_flag, &cb_stat));
        if (cb_flag) {
            OSHMPI_DBGMSG("received AMO packet origin %d, type %d\n",
                          cb_stat.MPI_SOURCE, am_pkt->type);

            /* Handle packet */
            int terminate_flag = 0;
            am_cb_handle_pkt(am_pkt, cb_stat.MPI_SOURCE, &terminate_flag);
            /* Do nothing when terminate. Last irecv will be completed at
             * progress_end. */

            /* Post next receive */
            OSHMPI_CALLMPI(MPI_Irecv(am_pkt, sizeof(OSHMPI_am_pkt_t),
                                     MPI_BYTE, MPI_ANY_SOURCE,
                                     OSHMPI_AM_PKT_TAG, OSHMPI_global.am_comm_world,
                                     &OSHMPI_global.am_req));
        }
    }

    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.am_cb_progress_cs);
}

OSHMPI_STATIC_INLINE_PREFIX void am_cb_progress_start(void)
{
    OSHMPI_am_pkt_t *am_pkt = (OSHMPI_am_pkt_t *) OSHMPI_global.am_pkt;

    /* Post first receive */
    OSHMPI_CALLMPI(MPI_Irecv(am_pkt, sizeof(OSHMPI_am_pkt_t),
                             MPI_BYTE, MPI_ANY_SOURCE,
                             OSHMPI_AM_PKT_TAG, OSHMPI_global.am_comm_world,
                             &OSHMPI_global.am_req));

#if defined(OSHMPI_ENABLE_AM_ASYNC_THREAD) || defined(OSHMPI_RUNTIME_AM_ASYNC_THREAD)
    if (OSHMPI_ENABLE_AM_ASYNC_THREAD_RUNTIME) {
        pthread_attr_t attr;

        OSHMPI_global.am_async_thread_done = 0;
        OSHMPI_CALLPTHREAD(pthread_mutex_init(&OSHMPI_global.am_async_mutex, NULL));
        OSHMPI_CALLPTHREAD(pthread_cond_init(&OSHMPI_global.am_async_cond, NULL));

        /* Create asynchronous progress thread */
        OSHMPI_CALLPTHREAD(pthread_attr_init(&attr));
        OSHMPI_CALLPTHREAD(pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED));
        OSHMPI_CALLPTHREAD(pthread_create
                           (&OSHMPI_global.am_async_thread, &attr, &am_cb_async_progress, NULL));
        OSHMPI_CALLPTHREAD(pthread_attr_destroy(&attr));
    }
#endif
}

OSHMPI_STATIC_INLINE_PREFIX void am_cb_progress_end(void)
{
    MPI_Status terminate_stat OSHMPI_ATTRIBUTE((unused));
    OSHMPI_am_pkt_t pkt;
    pkt.type = OSHMPI_AM_PKT_TERMINATE;

    /* Sent terminate signal to progress polling thread or consume the
     * last outstanding irecv on calling PE. */
    MPI_Send(&pkt, sizeof(OSHMPI_am_pkt_t), MPI_BYTE, OSHMPI_global.world_rank,
             OSHMPI_AM_PKT_TAG, OSHMPI_global.am_comm_world);
    OSHMPI_DBGMSG("sent terminate\n");

#if defined(OSHMPI_ENABLE_AM_ASYNC_THREAD) || defined(OSHMPI_RUNTIME_AM_ASYNC_THREAD)
    if (OSHMPI_ENABLE_AM_ASYNC_THREAD_RUNTIME) {
        OSHMPI_CALLPTHREAD(pthread_mutex_lock(&OSHMPI_global.am_async_mutex));
        while (!OSHMPI_global.am_async_thread_done) {
            OSHMPI_CALLPTHREAD(pthread_cond_wait
                               (&OSHMPI_global.am_async_cond, &OSHMPI_global.am_async_mutex));
        }
        OSHMPI_CALLPTHREAD(pthread_mutex_unlock(&OSHMPI_global.am_async_mutex));
        OSHMPI_CALLPTHREAD(pthread_cond_destroy(&OSHMPI_global.am_async_cond));
        OSHMPI_CALLPTHREAD(pthread_mutex_destroy(&OSHMPI_global.am_async_mutex));
        return; /* thread already completed last irecv */
    }
#endif

    /* Complete last irecv on calling PE. */
    OSHMPI_CALLMPI(MPI_Wait(&OSHMPI_global.am_req, &terminate_stat));
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_initialize(void)
{
    /* AM is unused if direct AMO is enabled */
    if (OSHMPI_ENABLE_DIRECT_AMO_RUNTIME)
        return;

    /* Dup comm world for the AMO progress thread */
    OSHMPI_CALLMPI(MPI_Comm_dup(OSHMPI_global.comm_world, &OSHMPI_global.am_comm_world));
    OSHMPI_CALLMPI(MPI_Comm_dup(OSHMPI_global.comm_world, &OSHMPI_global.am_ack_comm_world));

    /* Per PE flag indicating outstanding AM AMOs. */
    OSHMPI_global.am_outstanding_op_flags =
        OSHMPIU_malloc(sizeof(OSHMPI_atomic_flag_t) * OSHMPI_global.world_size);
    OSHMPI_ASSERT(OSHMPI_global.am_outstanding_op_flags);
    memset(OSHMPI_global.am_outstanding_op_flags, 0,
           sizeof(OSHMPI_atomic_flag_t) * OSHMPI_global.world_size);

    /* Global datatype table used for index translation */
    OSHMPI_global.am_datatypes_table =
        OSHMPIU_malloc(sizeof(MPI_Datatype) * OSHMPI_AM_MPI_DATATYPE_MAX);
    OSHMPI_ASSERT(OSHMPI_global.am_datatypes_table);
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_CHAR] = MPI_CHAR;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_SIGNED_CHAR] = MPI_SIGNED_CHAR;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_SHORT] = MPI_SHORT;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_INT] = MPI_INT;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_LONG] = MPI_LONG;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_LONG_LONG] = MPI_LONG_LONG;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_UNSIGNED_CHAR] = MPI_UNSIGNED_CHAR;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_UNSIGNED_SHORT] = MPI_UNSIGNED_SHORT;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_UNSIGNED] = MPI_UNSIGNED;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_UNSIGNED_LONG] = MPI_UNSIGNED_LONG;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_UNSIGNED_LONG_LONG] = MPI_UNSIGNED_LONG_LONG;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_INT8_T] = MPI_INT8_T;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_INT16_T] = MPI_INT16_T;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_INT32_T] = MPI_INT32_T;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_INT64_T] = MPI_INT64_T;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_UINT8_T] = MPI_UINT8_T;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_UINT16_T] = MPI_UINT16_T;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_UINT32_T] = MPI_UINT32_T;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_UINT64_T] = MPI_UINT64_T;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_OSHMPI_MPI_SIZE_T] = OSHMPI_MPI_SIZE_T;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_OSHMPI_MPI_PTRDIFF_T] = OSHMPI_MPI_PTRDIFF_T;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_FLOAT] = MPI_FLOAT;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_DOUBLE] = MPI_DOUBLE;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_LONG_DOUBLE] = MPI_LONG_DOUBLE;
    OSHMPI_global.am_datatypes_table[OSHMPI_AM_MPI_C_DOUBLE_COMPLEX] = MPI_C_DOUBLE_COMPLEX;

    /* Global op table used for index translation */
    OSHMPI_global.am_ops_table = OSHMPIU_malloc(sizeof(MPI_Op) * OSHMPI_AM_MPI_OP_MAX);
    OSHMPI_ASSERT(OSHMPI_global.am_ops_table);
    OSHMPI_global.am_ops_table[OSHMPI_AM_MPI_BAND] = MPI_BAND;
    OSHMPI_global.am_ops_table[OSHMPI_AM_MPI_BOR] = MPI_BOR;
    OSHMPI_global.am_ops_table[OSHMPI_AM_MPI_BXOR] = MPI_BXOR;
    OSHMPI_global.am_ops_table[OSHMPI_AM_MPI_NO_OP] = MPI_NO_OP;
    OSHMPI_global.am_ops_table[OSHMPI_AM_MPI_REPLACE] = MPI_REPLACE;
    OSHMPI_global.am_ops_table[OSHMPI_AM_MPI_SUM] = MPI_SUM;

    OSHMPI_global.am_pkt = OSHMPIU_malloc(sizeof(OSHMPI_am_pkt_t));
    OSHMPI_ASSERT(OSHMPI_global.am_pkt);

    OSHMPI_THREAD_INIT_CS(&OSHMPI_global.am_cb_progress_cs);
    am_cb_progress_start();

    OSHMPI_DBGMSG("Initialized active message AMO\n");
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_finalize(void)
{
    /* AM is unused if direct AMO is enabled */
    if (OSHMPI_ENABLE_DIRECT_AMO_RUNTIME)
        return;

    /* The finalize routine has to be called after implicity barrier
     * in shmem_finalize to ensure no incoming AMO request */

    am_cb_progress_end();
    OSHMPI_THREAD_DESTROY_CS(&OSHMPI_global.am_cb_progress_cs);

    OSHMPI_CALLMPI(MPI_Comm_free(&OSHMPI_global.am_comm_world));
    OSHMPI_CALLMPI(MPI_Comm_free(&OSHMPI_global.am_ack_comm_world));
    OSHMPIU_free(OSHMPI_global.am_outstanding_op_flags);
    OSHMPIU_free(OSHMPI_global.am_datatypes_table);
    OSHMPIU_free(OSHMPI_global.am_ops_table);
    OSHMPIU_free(OSHMPI_global.am_pkt);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_cb_progress(void)
{
    /* Skip progress if direct AMO is enabled */
    if (OSHMPI_ENABLE_DIRECT_AMO_RUNTIME)
        return;

    /* Make manual progress only when async thread is disabled */
    if (OSHMPI_ENABLE_AM_ASYNC_THREAD_RUNTIME)
        return;

    am_cb_manual_progress();
}

#endif /* INTERNAL_AM_IMPL_H */

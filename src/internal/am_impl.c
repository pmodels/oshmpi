/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#include "oshmpi_impl.h"

typedef struct {
    OSHMPI_am_cb_t func;
    char name[OSHMPI_AM_PKT_NAME_MAXLEN];       /* for debug message */
} OSHMPI_am_cb_regist_t;

static OSHMPI_am_cb_regist_t am_cb_funcs[OSHMPI_AM_PKT_MAX];

/* Callback of flush synchronization. */
void OSHMPI_am_flush_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt)
{
    OSHMPI_am_flush_pkt_t *flush_pkt = &pkt->flush;
    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Send(NULL, 0, MPI_BYTE, origin_rank, flush_pkt->ptag,
                            OSHMPI_global.am_ack_comm_world));
}

static void am_cb_handle_pkt(OSHMPI_am_pkt_t * am_pkt, int source_rank, int *terminate_flag)
{
    *terminate_flag = 0;

    /* Handle packet */
    switch (am_pkt->type) {
        case OSHMPI_AM_PKT_TERMINATE:
            OSHMPI_DBGMSG("received packet origin %d, type TERMINATE(%d)\n",
                          source_rank, am_pkt->type);
            *terminate_flag = 1;
            break;      /* Reached finalize */
        default:
            OSHMPI_ASSERT(am_pkt->type >= 0 && am_pkt->type < OSHMPI_AM_PKT_MAX &&
                          am_cb_funcs[am_pkt->type].func);
            OSHMPI_DBGMSG("received packet origin %d, type %s(%d)\n",
                          source_rank, am_cb_funcs[am_pkt->type].name, am_pkt->type);

            am_cb_funcs[am_pkt->type].func(source_rank, am_pkt);
            break;
    }
}

/* Async thread blocking progress polling for AMO active message */
#if !defined(OSHMPI_DISABLE_AM_ASYNC_THREAD)
static void *am_cb_async_progress(void *arg OSHMPI_ATTRIBUTE((unused)))
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
static void *am_cb_async_progress(void *arg OSHMPI_ATTRIBUTE((unused)))
{
    return NULL;
}
#endif /* end of !defined(OSHMPI_DISABLE_AM_ASYNC_THREAD) */

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

static void am_cb_progress_start(void)
{
    OSHMPI_am_pkt_t *am_pkt = (OSHMPI_am_pkt_t *) OSHMPI_global.am_pkt;

    /* Post first receive */
    OSHMPI_CALLMPI(MPI_Irecv(am_pkt, sizeof(OSHMPI_am_pkt_t),
                             MPI_BYTE, MPI_ANY_SOURCE,
                             OSHMPI_AM_PKT_TAG, OSHMPI_global.am_comm_world,
                             &OSHMPI_global.am_req));

#if !defined(OSHMPI_DISABLE_AM_ASYNC_THREAD)
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

static void am_cb_progress_end(void)
{
    MPI_Status terminate_stat OSHMPI_ATTRIBUTE((unused));
    OSHMPI_am_pkt_t pkt;
    pkt.type = OSHMPI_AM_PKT_TERMINATE;

    /* Sent terminate signal to progress polling thread or consume the
     * last outstanding irecv on calling PE. */
    MPI_Send(&pkt, sizeof(OSHMPI_am_pkt_t), MPI_BYTE, OSHMPI_global.world_rank,
             OSHMPI_AM_PKT_TAG, OSHMPI_global.am_comm_world);
    OSHMPI_DBGMSG("sent terminate\n");

#if !defined(OSHMPI_DISABLE_AM_ASYNC_THREAD)
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

void OSHMPI_am_cb_regist(OSHMPI_am_pkt_type_t pkt_type, const char *pkt_name,
                         OSHMPI_am_cb_t cb_func)
{
    OSHMPI_ASSERT(pkt_type >= 0 && pkt_type < OSHMPI_AM_PKT_MAX);
    am_cb_funcs[pkt_type].func = cb_func;
    if (pkt_name && strlen(pkt_name))
        strncpy(am_cb_funcs[pkt_type].name, pkt_name, OSHMPI_AM_PKT_NAME_MAXLEN - 1);

}

void OSHMPI_am_initialize(void)
{
    /* AM is not used if direct AMO is enabled and direct RMA is set at configure. */
    if (OSHMPI_ENABLE_DIRECT_AMO_RUNTIME && OSHMPI_ENABLE_DIRECT_RMA_CONFIG)
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

    int flag = 0, *am_pkt_ptag_ub_p = NULL;
    OSHMPI_ATOMIC_CNT_STORE(OSHMPI_global.am_pkt_ptag_off, 0);
    OSHMPI_CALLMPI(MPI_Comm_get_attr(OSHMPI_global.am_comm_world, MPI_TAG_UB,
                                     &am_pkt_ptag_ub_p, &flag));
    OSHMPI_ASSERT(flag);

    OSHMPI_global.am_pkt_ptag_ub = *am_pkt_ptag_ub_p;
    OSHMPI_ASSERT(OSHMPI_global.am_pkt_ptag_ub > OSHMPI_AM_PKT_PTAG_START);

    memset(am_cb_funcs, 0, sizeof(am_cb_funcs));

    OSHMPI_amo_am_initialize();
    OSHMPI_rma_am_initialize();
    OSHMPI_am_cb_regist(OSHMPI_AM_PKT_FLUSH, OSHMPI_am_flush_pkt_cb);

    OSHMPI_THREAD_INIT_CS(&OSHMPI_global.am_cb_progress_cs);
    am_cb_progress_start();

    OSHMPI_DBGMSG("Initialized active message AMO. am_pkt_ptag_ub %d\n",
                  OSHMPI_global.am_pkt_ptag_ub);
}

void OSHMPI_am_finalize(void)
{
    /* AM is not used if direct AMO is enabled and direct RMA is set at configure. */
    if (OSHMPI_ENABLE_DIRECT_AMO_RUNTIME && OSHMPI_ENABLE_DIRECT_RMA_CONFIG)
        return;

    /* The finalize routine has to be called after implicity barrier
     * in shmem_finalize to ensure no incoming AMO request */

    am_cb_progress_end();
    OSHMPI_THREAD_DESTROY_CS(&OSHMPI_global.am_cb_progress_cs);

    OSHMPI_amo_am_finalize();
    OSHMPI_rma_am_finalize();

    OSHMPI_CALLMPI(MPI_Comm_free(&OSHMPI_global.am_comm_world));
    OSHMPI_CALLMPI(MPI_Comm_free(&OSHMPI_global.am_ack_comm_world));
    OSHMPIU_free(OSHMPI_global.am_outstanding_op_flags);
    OSHMPIU_free(OSHMPI_global.am_datatypes_table);
    OSHMPIU_free(OSHMPI_global.am_ops_table);
    OSHMPIU_free(OSHMPI_global.am_pkt);
}

void OSHMPI_am_cb_progress(void)
{
    /* Make manual progress only when async thread is disabled */
    if (OSHMPI_ENABLE_AM_ASYNC_THREAD_RUNTIME)
        return;

    am_cb_manual_progress();
}

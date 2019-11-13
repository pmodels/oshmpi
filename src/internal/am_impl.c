/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include "oshmpi_impl.h"

static void poll_progress(int blocking, int *terminate_flag);

/* Active message and progress routines.
 * The progress mechanism relies on active message (e.g., the thread is
 * terminated by active message), thus we define them as a single component.*/

#if defined(OSHMPI_ENABLE_ASYNC_THREAD) || defined(OSHMPI_RUNTIME_ASYNC_THREAD)
static void *async_thread_fn(void *arg OSHMPI_ATTRIBUTE((unused)))
{
    int terminate_flag = 0;

    OSHMPI_DBGMSG("async thread started\n");

    poll_progress(OSHMPI_PROGRESS_BLOCKING, &terminate_flag);
    OSHMPI_ASSERT(terminate_flag == 1);

    OSHMPI_CALLPTHREAD(pthread_mutex_lock(&OSHMPI_global.async_mutex));
    OSHMPI_global.async_thread_done = 1;
    OSHMPI_CALLPTHREAD(pthread_mutex_unlock(&OSHMPI_global.async_mutex));
    OSHMPI_CALLPTHREAD(pthread_cond_signal(&OSHMPI_global.async_cond));

    OSHMPI_DBGMSG("async thread terminated\n");
    return NULL;
}
#endif /* defined(OSHMPI_ENABLE_ASYNC_THREAD) || defined(OSHMPI_RUNTIME_ASYNC_THREAD) */

static void initialize_progress(void)
{
#if defined(OSHMPI_ENABLE_ASYNC_THREAD) || defined(OSHMPI_RUNTIME_ASYNC_THREAD)
    if (OSHMPI_env.enable_async_thread) {
        pthread_attr_t attr;

        OSHMPI_global.async_thread_done = 0;
        OSHMPI_CALLPTHREAD(pthread_mutex_init(&OSHMPI_global.async_mutex, NULL));
        OSHMPI_CALLPTHREAD(pthread_cond_init(&OSHMPI_global.async_cond, NULL));

        /* Create asynchronous progress thread */
        OSHMPI_CALLPTHREAD(pthread_attr_init(&attr));
        OSHMPI_CALLPTHREAD(pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED));
        OSHMPI_CALLPTHREAD(pthread_create
                           (&OSHMPI_global.async_thread, &attr, &async_thread_fn, NULL));
        OSHMPI_CALLPTHREAD(pthread_attr_destroy(&attr));
    }
#endif
}

static void finalize_progress(void)
{
#if defined(OSHMPI_ENABLE_ASYNC_THREAD) || defined(OSHMPI_RUNTIME_ASYNC_THREAD)
    if (OSHMPI_env.enable_async_thread) {
        OSHMPI_pkt_t pkt;
        pkt.type = OSHMPI_PKT_TERMINATE;

        /* Sent terminate signal to progress polling thread */
        MPI_Send(&pkt, sizeof(OSHMPI_pkt_t), MPI_BYTE, OSHMPI_global.world_rank,
                 OSHMPI_PKT_TAG, OSHMPI_global.am_comm_world);

        OSHMPI_CALLPTHREAD(pthread_mutex_lock(&OSHMPI_global.async_mutex));
        while (!OSHMPI_global.async_thread_done) {
            OSHMPI_CALLPTHREAD(pthread_cond_wait
                               (&OSHMPI_global.async_cond, &OSHMPI_global.async_mutex));
        }
        OSHMPI_CALLPTHREAD(pthread_mutex_unlock(&OSHMPI_global.async_mutex));
        OSHMPI_CALLPTHREAD(pthread_cond_destroy(&OSHMPI_global.async_cond));
        OSHMPI_CALLPTHREAD(pthread_mutex_destroy(&OSHMPI_global.async_mutex));
        return;
    }
#endif
}

#define OSHMPI_PROGRESS_POLL_NCNT 1

static void poll_progress(int blocking, int *terminate_flag)
{
    OSHMPI_pkt_t *am_pkt = (OSHMPI_pkt_t *) OSHMPI_global.am_pkt;
    int poll_cnt = OSHMPI_PROGRESS_POLL_NCNT;

    /* Only one thread can poll progress at a time */
    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.am_progress_cs);

    /* Use am_comm_world to send/receive AMO packets from remote PEs
     * or termination flag from the main thread; use amo_ack_comm_world
     * for ack send in callback in order to avoid interaction with AMO packets.
     * Fetch or flush issuing routine can directly receive the ack
     * without going through the callback progress. */
    while (blocking == OSHMPI_PROGRESS_BLOCKING || poll_cnt-- > 0) {
        int cb_flag = 0;
        MPI_Status cb_stat;

        OSHMPI_CALLMPI(MPI_Test(&OSHMPI_global.am_req, &cb_flag, &cb_stat));
        if (cb_flag) {
            OSHMPI_global.am_req = MPI_REQUEST_NULL;
            OSHMPI_DBGMSG("received packet origin %d, type %d\n", cb_stat.MPI_SOURCE, am_pkt->type);
            /* Handle packet */
            switch (am_pkt->type) {
#if !defined(OSHMPI_ENABLE_DIRECT_AMO)  /* enable AM_AMO or AUTO */
                case OSHMPI_PKT_AMO_CSWAP:
                    OSHMPI_amo_cswap_pkt_cb(cb_stat.MPI_SOURCE, am_pkt);
                    break;
                case OSHMPI_PKT_AMO_FETCH:
                    OSHMPI_amo_fetch_pkt_cb(cb_stat.MPI_SOURCE, am_pkt);
                    break;
                case OSHMPI_PKT_AMO_POST:
                    OSHMPI_amo_post_pkt_cb(cb_stat.MPI_SOURCE, am_pkt);
                    break;
                case OSHMPI_PKT_AMO_FLUSH:
                    OSHMPI_amo_flush_pkt_cb(cb_stat.MPI_SOURCE, am_pkt);
                    break;
#endif
                case OSHMPI_PKT_TERMINATE:
                    OSHMPI_DBGMSG("received terminate\n");
                    if (terminate_flag)
                        *terminate_flag = 1;
                    goto fn_exit;       /* The main thread reached finalize */
                    break;
                default:
                    OSHMPI_ERR_ABORT("Unsupported AM packet type: %d\n", am_pkt->type);
                    break;
            }

            /* Post next receive */
            OSHMPI_CALLMPI(MPI_Irecv(am_pkt, sizeof(OSHMPI_pkt_t),
                                     MPI_BYTE, MPI_ANY_SOURCE,
                                     MPI_ANY_TAG, OSHMPI_global.am_comm_world,
                                     &OSHMPI_global.am_req));
        }
        /* THREAD_YIELD */
    }

  fn_exit:
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.am_progress_cs);
    return;
}


void OSHMPI_am_initialize(void)
{
    /* Dup comm world for AM */
    OSHMPI_CALLMPI(MPI_Comm_dup(OSHMPI_global.comm_world, &OSHMPI_global.am_comm_world));

    OSHMPI_global.am_req = MPI_REQUEST_NULL;
    OSHMPI_global.am_pkt = OSHMPIU_malloc(sizeof(OSHMPI_pkt_t));
    OSHMPI_ASSERT(OSHMPI_global.am_pkt);

    OSHMPI_THREAD_INIT_CS(&OSHMPI_global.am_progress_cs);

    /* Post first receive */
    OSHMPI_CALLMPI(MPI_Irecv(OSHMPI_global.am_pkt, sizeof(OSHMPI_pkt_t),
                             MPI_BYTE, MPI_ANY_SOURCE, OSHMPI_PKT_TAG,
                             OSHMPI_global.am_comm_world, &OSHMPI_global.am_req));

    initialize_progress();

    OSHMPI_DBGMSG("Initialized active message\n");
}

void OSHMPI_am_finalize(void)
{
    /* The finalize routine has to be called after implicity barrier
     * in shmem_finalize to ensure no incoming AM request */

    finalize_progress();

    /* Ensure last receive is completed on calling PE.
     * A null request means that the last receive was already completed
     * by the async thread.*/
    if (OSHMPI_global.am_req != MPI_REQUEST_NULL) {
        OSHMPI_pkt_t pkt;
        pkt.type = OSHMPI_PKT_TERMINATE;

        MPI_Send(&pkt, sizeof(OSHMPI_pkt_t), MPI_BYTE, OSHMPI_global.world_rank,
                 OSHMPI_PKT_TAG, OSHMPI_global.am_comm_world);
        OSHMPI_CALLMPI(MPI_Wait(&OSHMPI_global.am_req, MPI_STATUS_IGNORE));
    }

    OSHMPI_THREAD_DESTROY_CS(&OSHMPI_global.am_progress_cs);

    OSHMPI_CALLMPI(MPI_Comm_free(&OSHMPI_global.am_comm_world));
    OSHMPIU_free(OSHMPI_global.am_pkt);

    OSHMPI_DBGMSG("Finalized active message\n");
}

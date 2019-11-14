/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include "oshmpi_impl.h"

#if !defined(OSHMPI_ENABLE_AM_AMO)      /* direct or runtime */
static void amo_direct_initialize(void)
{
    OSHMPI_DBGMSG("Initialized direct AMO\n");
}

static void amo_direct_finalize(void)
{
}
#endif /* End of !defined(OSHMPI_ENABLE_AM_AMO) */

#if !defined(OSHMPI_ENABLE_DIRECT_AMO)  /* am or runtime */
static void amo_am_initialize(void)
{
    /* Dup comm world for the AMO ACK  */
    OSHMPI_CALLMPI(MPI_Comm_dup(OSHMPI_global.comm_world, &OSHMPI_global.amo_ack_comm_world));

    /* Per PE flag indicating outstanding AM AMOs. */
    OSHMPI_global.amo_outstanding_op_flags =
        OSHMPIU_malloc(sizeof(OSHMPI_atomic_flag_t) * OSHMPI_global.world_size);
    OSHMPI_ASSERT(OSHMPI_global.amo_outstanding_op_flags);
    memset(OSHMPI_global.amo_outstanding_op_flags, 0,
           sizeof(OSHMPI_atomic_flag_t) * OSHMPI_global.world_size);

    OSHMPI_am_cb_regist(OSHMPI_PKT_AMO_CSWAP, "AMO_CSWAP", OSHMPI_amo_cswap_pkt_cb);
    OSHMPI_am_cb_regist(OSHMPI_PKT_AMO_FETCH, "AMO_FETCH", OSHMPI_amo_fetch_pkt_cb);
    OSHMPI_am_cb_regist(OSHMPI_PKT_AMO_POST, "AMO_POST", OSHMPI_amo_post_pkt_cb);
    OSHMPI_am_cb_regist(OSHMPI_PKT_AMO_FLUSH, "AMO_FLUSH", OSHMPI_amo_flush_pkt_cb);

    OSHMPI_DBGMSG("Initialized active message AMO\n");
}

static void amo_am_finalize(void)
{
    OSHMPI_CALLMPI(MPI_Comm_free(&OSHMPI_global.amo_ack_comm_world));
    OSHMPIU_free(OSHMPI_global.amo_outstanding_op_flags);
}
#endif /* End of !defined(OSHMPI_ENABLE_DIRECT_AMO) */

void OSHMPI_amo_initialize(void)
{
#if defined(OSHMPI_ENABLE_DIRECT_AMO)
    amo_direct_initialize();
#elif defined(OSHMPI_ENABLE_AM_AMO)
    amo_am_initialize();
#else /* Default make decision at runtime */
    if (OSHMPI_global.amo_direct)
        amo_direct_initialize();
    else
        amo_am_initialize();
#endif
}

void OSHMPI_amo_finalize(void)
{
#if defined(OSHMPI_ENABLE_DIRECT_AMO)
    amo_direct_finalize();
#elif defined(OSHMPI_ENABLE_AM_AMO)
    amo_am_finalize();
#else /* Default make decision at runtime */
    if (OSHMPI_global.amo_direct)
        amo_direct_finalize();
    else
        amo_am_finalize();
#endif
}

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
    OSHMPIU_free(OSHMPI_global.amo_datatypes_table);
    OSHMPIU_free(OSHMPI_global.amo_ops_table);
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

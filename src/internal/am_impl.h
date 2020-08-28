/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_AM_IMPL_H
#define INTERNAL_AM_IMPL_H

#include "oshmpi_impl.h"

/* Get unique package tag for active messages of each operation in order to
 * avoid mismatch in multithreading. Always requested by the operation initiator.
 * Note that we require only unique ID per peer as the consequent messages always
 * have specific source rank. However, here we use an atomic counter per PE for simplicity. */
OSHMPI_STATIC_INLINE_PREFIX int OSHMPI_am_get_pkt_ptag(void)
{
    int tag = OSHMPI_ATOMIC_CNT_FINC(OSHMPI_global.am_pkt_ptag_off) + OSHMPI_AM_PKT_PTAG_START;
    OSHMPI_ASSERT(tag < OSHMPI_global.am_pkt_ptag_ub);
    return tag;
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_flush(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                 int PE_start, int logPE_stride, int PE_size)
{
    OSHMPI_am_pkt_t pkt;
    OSHMPI_am_flush_pkt_t *flush_pkt = &pkt.flush;
    int pe, nreqs = 0, noutstanding_pes = 0, i;
    MPI_Request *reqs = NULL;
    const int pe_stride = 1 << logPE_stride;    /* Implement 2^pe_logs with bitshift. */

    /* No AM flush is needed if direct AMO is enabled and
     * direct RMA is set at configure. */
    if (OSHMPI_ENABLE_DIRECT_AMO_RUNTIME && OSHMPI_ENABLE_DIRECT_RMA_CONFIG)
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
    flush_pkt->ptag = OSHMPI_am_get_pkt_ptag();

    for (i = 0; i < PE_size; i++) {
        pe = PE_start + i * pe_stride;
        if (OSHMPI_ATOMIC_FLAG_LOAD(OSHMPI_global.am_outstanding_op_flags[pe])) {
            OSHMPI_CALLMPI(MPI_Isend
                           (&pkt, sizeof(OSHMPI_am_pkt_t), MPI_BYTE, pe, OSHMPI_AM_PKT_TAG,
                            OSHMPI_global.am_comm_world, &reqs[nreqs++]));
            OSHMPI_CALLMPI(MPI_Irecv(NULL, 0, MPI_BYTE, pe, flush_pkt->ptag,
                                     OSHMPI_global.am_ack_comm_world, &reqs[nreqs++]));
            OSHMPI_DBGMSG("packet type %d, target %d in [start %d, stride %d, size %d], ptag %d\n",
                          pkt.type, pe, PE_start, logPE_stride, PE_size, flush_pkt->ptag);
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
    /* No AM flush is needed if direct AMO is enabled and
     * direct RMA is set at configure. */
    if (OSHMPI_ENABLE_DIRECT_AMO_RUNTIME && OSHMPI_ENABLE_DIRECT_RMA_CONFIG)
        return;

    OSHMPI_am_flush(ctx, 0 /* PE_start */ , 0 /* logPE_stride */ ,
                    OSHMPI_global.world_size /* PE_size */);
}

#endif /* INTERNAL_AM_IMPL_H */

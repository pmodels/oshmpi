/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#ifndef INTERNAL_AMO_AM_IMPL_H
#define INTERNAL_AMO_AM_IMPL_H

#include "oshmpi_impl.h"

/* Issue a compare_and_swap operation. Blocking wait until return of old value. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_cswap(shmem_ctx_t ctx
                                                     OSHMPI_ATTRIBUTE((unused)),
                                                     MPI_Datatype mpi_type,
                                                     OSHMPI_amo_mpi_datatype_index_t mpi_type_idx,
                                                     size_t bytes, void *dest, void *cond_ptr,
                                                     void *value_ptr, int pe, void *oldval_ptr)
{
    OSHMPI_pkt_t pkt;
    OSHMPI_amo_cswap_pkt_t *cswap_pkt = &pkt.cswap;
    MPI_Aint target_disp = -1;
    MPI_Win win = MPI_WIN_NULL;

    OSHMPI_translate_win_and_disp((const void *) dest, &win, &target_disp);
    OSHMPI_ASSERT(target_disp >= 0 && win != MPI_WIN_NULL);

    pkt.type = OSHMPI_PKT_AMO_CSWAP;
    memcpy(&cswap_pkt->cond, cond_ptr, bytes);
    memcpy(&cswap_pkt->value, value_ptr, bytes);
    cswap_pkt->target_disp = target_disp;
    cswap_pkt->mpi_type_idx = mpi_type_idx;
    cswap_pkt->bytes = bytes;
    cswap_pkt->symm_obj_type = (win == OSHMPI_global.symm_heap_win) ?
        OSHMPI_SYMM_OBJ_HEAP : OSHMPI_SYMM_OBJ_DATA;

    OSHMPI_CALLMPI(MPI_Send(&pkt, sizeof(OSHMPI_pkt_t), MPI_BYTE, pe, OSHMPI_PKT_TAG,
                            OSHMPI_global.am_comm_world));

    OSHMPI_CALLMPI(MPI_Recv(oldval_ptr, 1, mpi_type, pe, OSHMPI_PKT_AMO_ACK_TAG,
                            OSHMPI_global.amo_ack_comm_world, MPI_STATUS_IGNORE));

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
    OSHMPI_pkt_t pkt;
    OSHMPI_amo_fetch_pkt_t *fetch_pkt = &pkt.fetch;
    MPI_Aint target_disp = -1;
    MPI_Win win = MPI_WIN_NULL;

    OSHMPI_translate_win_and_disp((const void *) dest, &win, &target_disp);
    OSHMPI_ASSERT(target_disp >= 0 && win != MPI_WIN_NULL);

    pkt.type = OSHMPI_PKT_AMO_FETCH;
    fetch_pkt->target_disp = target_disp;
    fetch_pkt->mpi_type_idx = mpi_type_idx;
    fetch_pkt->mpi_op_idx = op_idx;
    fetch_pkt->bytes = bytes;
    if (fetch_pkt->mpi_op_idx != OSHMPI_AMO_MPI_NO_OP)
        memcpy(&fetch_pkt->value, value_ptr, bytes);    /* ignore value in atomic-fetch */
    fetch_pkt->symm_obj_type = (win == OSHMPI_global.symm_heap_win) ?
        OSHMPI_SYMM_OBJ_HEAP : OSHMPI_SYMM_OBJ_DATA;

    OSHMPI_CALLMPI(MPI_Send(&pkt, sizeof(OSHMPI_pkt_t), MPI_BYTE, pe, OSHMPI_PKT_TAG,
                            OSHMPI_global.am_comm_world));

    OSHMPI_CALLMPI(MPI_Recv(oldval_ptr, 1, mpi_type, pe, OSHMPI_PKT_AMO_ACK_TAG,
                            OSHMPI_global.amo_ack_comm_world, MPI_STATUS_IGNORE));

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
    OSHMPI_pkt_t pkt;
    OSHMPI_amo_post_pkt_t *post_pkt = &pkt.post;
    MPI_Aint target_disp = -1;
    MPI_Win win = MPI_WIN_NULL;

    OSHMPI_translate_win_and_disp((const void *) dest, &win, &target_disp);
    OSHMPI_ASSERT(target_disp >= 0 && win != MPI_WIN_NULL);

    pkt.type = OSHMPI_PKT_AMO_POST;
    post_pkt->target_disp = target_disp;
    post_pkt->mpi_type_idx = mpi_type_idx;
    post_pkt->mpi_op_idx = op_idx;
    post_pkt->bytes = bytes;
    memcpy(&post_pkt->value, value_ptr, bytes);
    post_pkt->symm_obj_type = (win == OSHMPI_global.symm_heap_win) ?
        OSHMPI_SYMM_OBJ_HEAP : OSHMPI_SYMM_OBJ_DATA;

    OSHMPI_CALLMPI(MPI_Send(&pkt, sizeof(OSHMPI_pkt_t), MPI_BYTE, pe, OSHMPI_PKT_TAG,
                            OSHMPI_global.am_comm_world));

    OSHMPI_DBGMSG("packet type %d, symm type %s, target %d, datatype idx %d, op idx %d\n",
                  pkt.type, (post_pkt->symm_obj_type == OSHMPI_SYMM_OBJ_HEAP) ? "heap" : "data",
                  pe, mpi_type_idx, op_idx);

    /* Indicate outstanding AMO */
    OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_global.amo_outstanding_op_flags[pe], 1);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_flush(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                                     int PE_start, int logPE_stride, int PE_size)
{
    OSHMPI_pkt_t pkt;
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
    pkt.type = OSHMPI_PKT_AMO_FLUSH;

    for (i = 0; i < PE_size; i++) {
        pe = PE_start + i * pe_stride;
        if (OSHMPI_ATOMIC_FLAG_LOAD(OSHMPI_global.amo_outstanding_op_flags[pe])) {
            OSHMPI_CALLMPI(MPI_Isend
                           (&pkt, sizeof(OSHMPI_pkt_t), MPI_BYTE, pe, OSHMPI_PKT_TAG,
                            OSHMPI_global.am_comm_world, &reqs[nreqs++]));
            OSHMPI_CALLMPI(MPI_Irecv
                           (NULL, 0, MPI_BYTE, pe, OSHMPI_PKT_AMO_ACK_TAG,
                            OSHMPI_global.amo_ack_comm_world, &reqs[nreqs++]));
            OSHMPI_DBGMSG("packet type %d, target %d in [start %d, stride %d, size %d]\n",
                          pkt.type, pe, PE_start, logPE_stride, PE_size);
        }
    }

    OSHMPI_ASSERT(PE_size * 2 >= nreqs);
    OSHMPI_CALLMPI(MPI_Waitall(nreqs, reqs, MPI_STATUS_IGNORE));
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

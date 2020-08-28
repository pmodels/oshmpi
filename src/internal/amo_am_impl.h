/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_AMO_AM_IMPL_H
#define INTERNAL_AMO_AM_IMPL_H

#include "oshmpi_impl.h"

/* Issue a compare_and_swap operation. Blocking wait until return of old value. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_cswap(OSHMPI_ictx_t * ictx,
                                                     MPI_Datatype mpi_type,
                                                     OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                     size_t bytes, void *dest, void *cond_ptr,
                                                     void *value_ptr, int pe, void *oldval_ptr,
                                                     OSHMPI_sobj_attr_t * sobj_attr)
{
    OSHMPI_am_pkt_t pkt;
    OSHMPI_am_cswap_pkt_t *cswap_pkt = &pkt.cswap;

    pkt.type = OSHMPI_AM_PKT_CSWAP;
    memcpy(&cswap_pkt->cond, cond_ptr, bytes);
    memcpy(&cswap_pkt->value, value_ptr, bytes);
    cswap_pkt->mpi_type_idx = mpi_type_idx;
    cswap_pkt->bytes = bytes;
    cswap_pkt->sobj_handle = sobj_attr->handle;
    cswap_pkt->ptag = OSHMPI_am_get_pkt_ptag();

    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, (const void *) dest, pe, OSHMPI_RELATIVE_DISP,
                                    &cswap_pkt->target_disp);
    OSHMPI_ASSERT(cswap_pkt->target_disp >= 0);

    OSHMPI_am_progress_mpi_send(&pkt, sizeof(OSHMPI_am_pkt_t), MPI_BYTE, pe, OSHMPI_AM_PKT_TAG,
                                OSHMPI_am.comm);

    OSHMPI_am_progress_mpi_recv(oldval_ptr, 1, mpi_type, pe, cswap_pkt->ptag,
                                OSHMPI_am.ack_comm, MPI_STATUS_IGNORE);

    OSHMPI_DBGMSG
        ("packet type %d, sobj_handle 0x%x, target %d, datatype idx %d, addr %p, disp 0x%lx, ptag %d\n",
         pkt.type, cswap_pkt->sobj_handle, pe, mpi_type_idx, dest, cswap_pkt->target_disp,
         cswap_pkt->ptag);

    /* Reset flag since remote PE should have finished previous post
     * before handling this fetch. */
    OSHMPIU_ATOMIC_FLAG_STORE(OSHMPI_am.outstanding_op_flags[pe], 0);
}

/* Issue a fetch (with op) operation. Blocking wait until return of old value. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_fetch(OSHMPI_ictx_t * ictx,
                                                     MPI_Datatype mpi_type,
                                                     OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                     size_t bytes, MPI_Op op,
                                                     OSHMPI_am_mpi_op_index_t op_idx, void *dest,
                                                     void *value_ptr, int pe, void *oldval_ptr,
                                                     OSHMPI_sobj_attr_t * sobj_attr)
{
    OSHMPI_am_pkt_t pkt;
    OSHMPI_am_fetch_pkt_t *fetch_pkt = &pkt.fetch;

    pkt.type = OSHMPI_AM_PKT_FETCH;
    fetch_pkt->mpi_type_idx = mpi_type_idx;
    fetch_pkt->mpi_op_idx = op_idx;
    fetch_pkt->bytes = bytes;
    fetch_pkt->sobj_handle = sobj_attr->handle;
    fetch_pkt->ptag = OSHMPI_am_get_pkt_ptag();

    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, (const void *) dest, pe, OSHMPI_RELATIVE_DISP,
                                    &fetch_pkt->target_disp);
    OSHMPI_ASSERT(fetch_pkt->target_disp >= 0);

    if (fetch_pkt->mpi_op_idx != OSHMPI_AM_MPI_NO_OP)
        memcpy(&fetch_pkt->value, value_ptr, bytes);    /* ignore value in atomic-fetch */

    OSHMPI_am_progress_mpi_send(&pkt, sizeof(OSHMPI_am_pkt_t), MPI_BYTE, pe, OSHMPI_AM_PKT_TAG,
                                OSHMPI_am.comm);

    OSHMPI_am_progress_mpi_recv(oldval_ptr, 1, mpi_type, pe, fetch_pkt->ptag,
                                OSHMPI_am.ack_comm, MPI_STATUS_IGNORE);

    OSHMPI_DBGMSG
        ("packet type %d, sobj_handle 0x%x, target %d, datatype idx %d, op idx %d, addr %p, disp 0x%lx, ptag %d\n",
         pkt.type, fetch_pkt->sobj_handle, pe, mpi_type_idx, op_idx, dest, fetch_pkt->target_disp,
         fetch_pkt->ptag);

    /* Reset flag since remote PE should have finished previous post
     * before handling this fetch. */
    OSHMPIU_ATOMIC_FLAG_STORE(OSHMPI_am.outstanding_op_flags[pe], 0);
}

/* Issue a post operation. Return immediately after sent AMO packet */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_post(OSHMPI_ictx_t * ictx,
                                                    MPI_Datatype mpi_type,
                                                    OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                    size_t bytes, MPI_Op op,
                                                    OSHMPI_am_mpi_op_index_t op_idx, void *dest,
                                                    void *value_ptr, int pe,
                                                    OSHMPI_sobj_attr_t * sobj_attr)
{
    OSHMPI_am_pkt_t pkt;
    OSHMPI_am_post_pkt_t *post_pkt = &pkt.post;

    pkt.type = OSHMPI_AM_PKT_POST;
    post_pkt->mpi_type_idx = mpi_type_idx;
    post_pkt->mpi_op_idx = op_idx;
    post_pkt->bytes = bytes;
    post_pkt->sobj_handle = sobj_attr->handle;
    memcpy(&post_pkt->value, value_ptr, bytes);

    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, (const void *) dest, pe, OSHMPI_RELATIVE_DISP,
                                    &post_pkt->target_disp);
    OSHMPI_ASSERT(post_pkt->target_disp >= 0);

    OSHMPI_am_progress_mpi_send(&pkt, sizeof(OSHMPI_am_pkt_t), MPI_BYTE, pe, OSHMPI_AM_PKT_TAG,
                                OSHMPI_am.comm);
    OSHMPI_DBGMSG
        ("packet type %d, sobj_handle 0x%x, target %d, datatype idx %d, op idx %d, addr %p, disp 0x%lx\n",
         pkt.type, post_pkt->sobj_handle, pe, mpi_type_idx, op_idx, dest, post_pkt->target_disp);

    /* Indicate outstanding AMO */
    OSHMPIU_ATOMIC_FLAG_STORE(OSHMPI_am.outstanding_op_flags[pe], 1);
}

#endif /* INTERNAL_AMO_AM_IMPL_H */

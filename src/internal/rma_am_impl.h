/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#ifndef INTERNAL_RMA_AM_IMPL_H
#define INTERNAL_RMA_AM_IMPL_H

#include "oshmpi_impl.h"

/* Issue a PUT operation. Return immediately after sent PUT packet (local complete) */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_rma_am_put(OSHMPI_ictx_t * ictx,
                                                   MPI_Datatype mpi_type, size_t typesz,
                                                   const void *origin_addr, void *target_addr,
                                                   size_t nelems, int pe,
                                                   OSHMPI_sobj_attr_t * sobj_attr)
{
    OSHMPI_am_pkt_t pkt;
    OSHMPI_am_put_pkt_t *put_pkt = &pkt.put;

    pkt.type = OSHMPI_AM_PKT_PUT;
    put_pkt->bytes = typesz * nelems;
    put_pkt->sobj_handle = sobj_attr->handle;
    put_pkt->ptag = OSHMPI_am_get_pkt_ptag();

    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, (const void *) target_addr, pe, OSHMPI_RELATIVE_DISP,
                                    &put_pkt->target_disp);
    OSHMPI_ASSERT(put_pkt->target_disp >= 0);

    OSHMPI_am_progress_mpi_send(&pkt, sizeof(OSHMPI_am_pkt_t), MPI_BYTE, pe, OSHMPI_AM_PKT_TAG,
                                OSHMPI_am.comm);

    OSHMPI_am_progress_mpi_send(origin_addr, nelems, mpi_type, pe, put_pkt->ptag, OSHMPI_am.comm);
    OSHMPI_DBGMSG
        ("packet type %d, sobj_handle 0x%x, target %d, bytes %ld, addr %p, disp 0x%lx, ptag %d\n",
         pkt.type, put_pkt->sobj_handle, pe, put_pkt->bytes, target_addr, put_pkt->target_disp,
         put_pkt->ptag);

    /* Indicate outstanding AM */
    OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_am.outstanding_op_flags[pe], 1);
}

/* Issue a GET operation. Return after receiving return value. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_rma_am_get(OSHMPI_ictx_t * ictx,
                                                   MPI_Datatype mpi_type, size_t typesz,
                                                   void *origin_addr, const void *target_addr,
                                                   size_t nelems, int pe,
                                                   OSHMPI_sobj_attr_t * sobj_attr)
{
    OSHMPI_am_pkt_t pkt;
    OSHMPI_am_get_pkt_t *get_pkt = &pkt.get;

    pkt.type = OSHMPI_AM_PKT_GET;
    get_pkt->bytes = typesz * nelems;
    get_pkt->sobj_handle = sobj_attr->handle;
    get_pkt->ptag = OSHMPI_am_get_pkt_ptag();

    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, target_addr, pe, OSHMPI_RELATIVE_DISP,
                                    &get_pkt->target_disp);
    OSHMPI_ASSERT(get_pkt->target_disp >= 0);

    OSHMPI_am_progress_mpi_send(&pkt, sizeof(OSHMPI_am_pkt_t), MPI_BYTE, pe, OSHMPI_AM_PKT_TAG,
                                OSHMPI_am.comm);

    OSHMPI_am_progress_mpi_recv(origin_addr, nelems, mpi_type, pe, get_pkt->ptag,
                                OSHMPI_am.ack_comm, MPI_STATUS_IGNORE);

    OSHMPI_DBGMSG
        ("packet type %d, sobj_handle 0x%x, target %d, bytes %ld, addr %p, disp 0x%lx, ptag %d\n",
         pkt.type, get_pkt->sobj_handle, pe, get_pkt->bytes, target_addr, get_pkt->target_disp,
         get_pkt->ptag);

    /* Reset flag since remote PE should have finished previous put
     * before handling this get. */
    OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_am.outstanding_op_flags[pe], 0);
}

/* Issue a strided PUT operation. Return immediately after sent PUT packet (local complete) */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_rma_am_iput(OSHMPI_ictx_t * ictx,
                                                    MPI_Datatype mpi_type,
                                                    OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                    const void *origin_addr, void *target_addr,
                                                    ptrdiff_t origin_st, ptrdiff_t target_st,
                                                    size_t nelems, int pe,
                                                    OSHMPI_sobj_attr_t * sobj_attr)
{
    OSHMPI_am_pkt_t pkt;
    OSHMPI_am_iput_pkt_t *iput_pkt = &pkt.iput;

    pkt.type = OSHMPI_AM_PKT_IPUT;
    iput_pkt->mpi_type_idx = mpi_type_idx;
    iput_pkt->target_st = target_st;
    iput_pkt->nelems = nelems;
    iput_pkt->sobj_handle = sobj_attr->handle;
    iput_pkt->ptag = OSHMPI_am_get_pkt_ptag();

    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, (const void *) target_addr, pe, OSHMPI_RELATIVE_DISP,
                                    &iput_pkt->target_disp);
    OSHMPI_ASSERT(iput_pkt->target_disp >= 0);

    OSHMPI_am_progress_mpi_send(&pkt, sizeof(OSHMPI_am_pkt_t), MPI_BYTE, pe, OSHMPI_AM_PKT_TAG,
                                OSHMPI_am.comm);

    MPI_Datatype origin_type = MPI_DATATYPE_NULL;
    size_t origin_count = 0;
    OSHMPI_create_strided_dtype(nelems, origin_st, mpi_type, 0 /* no required extent */ ,
                                &origin_count, &origin_type);

    OSHMPI_am_progress_mpi_send(origin_addr, origin_count, origin_type, pe, iput_pkt->ptag,
                                OSHMPI_am.comm);
    OSHMPI_DBGMSG("packet type %d, sobj_handle 0x%x, target %d, datatype idx %d, "
                  "origin_st 0x%lx, target_st 0x%lx, nelems %ld, addr %p, disp 0x%lx, ptag %d\n",
                  pkt.type, iput_pkt->sobj_handle, pe, mpi_type_idx, origin_st, target_st, nelems,
                  target_addr, iput_pkt->target_disp, iput_pkt->ptag);

    /* Indicate outstanding AM */
    OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_am.outstanding_op_flags[pe], 1);
    OSHMPI_free_strided_dtype(mpi_type, &origin_type);
}

/* Issue a strided GET operation. Return after receiving return value. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_rma_am_iget(OSHMPI_ictx_t * ictx,
                                                    MPI_Datatype mpi_type,
                                                    OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                    void *origin_addr, const void *target_addr,
                                                    ptrdiff_t origin_st, ptrdiff_t target_st,
                                                    size_t nelems, int pe,
                                                    OSHMPI_sobj_attr_t * sobj_attr)
{
    OSHMPI_am_pkt_t pkt;
    OSHMPI_am_iget_pkt_t *iget_pkt = &pkt.iget;

    pkt.type = OSHMPI_AM_PKT_IGET;
    iget_pkt->mpi_type_idx = mpi_type_idx;
    iget_pkt->target_st = target_st;
    iget_pkt->nelems = nelems;
    iget_pkt->sobj_handle = sobj_attr->handle;
    iget_pkt->ptag = OSHMPI_am_get_pkt_ptag();

    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, target_addr, pe, OSHMPI_RELATIVE_DISP,
                                    &iget_pkt->target_disp);
    OSHMPI_ASSERT(iget_pkt->target_disp >= 0);

    OSHMPI_am_progress_mpi_send(&pkt, sizeof(OSHMPI_am_pkt_t), MPI_BYTE, pe, OSHMPI_AM_PKT_TAG,
                                OSHMPI_am.comm);

    MPI_Datatype origin_type = MPI_DATATYPE_NULL;
    size_t origin_count = 0;
    OSHMPI_create_strided_dtype(nelems, origin_st, mpi_type, 0 /* no required extent */ ,
                                &origin_count, &origin_type);

    OSHMPI_am_progress_mpi_recv(origin_addr, origin_count, origin_type, pe, iget_pkt->ptag,
                                OSHMPI_am.ack_comm, MPI_STATUS_IGNORE);

    OSHMPI_DBGMSG("packet type %d, sobj_handle 0x%x, target %d, datatype idx %d, "
                  "origin_st 0x%lx, target_st 0x%lx, nelems %ld, addr %p, disp 0x%lx, ptag %d\n",
                  pkt.type, iget_pkt->sobj_handle, pe, mpi_type_idx, origin_st, target_st, nelems,
                  target_addr, iget_pkt->target_disp, iget_pkt->ptag);

    /* Reset flag since remote PE should have finished previous put
     * before handling this get. */
    OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_am.outstanding_op_flags[pe], 0);
    OSHMPI_free_strided_dtype(mpi_type, &origin_type);
}

#endif /* INTERNAL_RMA_AM_IMPL_H */

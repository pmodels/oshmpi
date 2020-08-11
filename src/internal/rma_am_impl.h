/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#ifndef INTERNAL_RMA_AM_IMPL_H
#define INTERNAL_RMA_AM_IMPL_H

#include "oshmpi_impl.h"

/* Callback of PUT operation. Receive data to local symm object.
 * No ACK is returned to origin PE. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_rma_am_put_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt)
{
    void *dest = NULL;
    OSHMPI_am_put_pkt_t *put_pkt = &pkt->put;

    OSHMPI_translate_disp_to_vaddr(put_pkt->sobj_handle, put_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Recv(dest, put_pkt->bytes, MPI_BYTE, origin_rank,
                            OSHMPI_AM_PKT_DATA_TAG, OSHMPI_global.am_comm_world,
                            MPI_STATUS_IGNORE));
}

/* Callback of GET operation. Send data from local symm object to origin PE. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_rma_am_get_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt)
{
    OSHMPI_am_get_pkt_t *get_pkt = &pkt->get;
    void *dest = NULL;

    OSHMPI_translate_disp_to_vaddr(get_pkt->sobj_handle, get_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Send(dest, get_pkt->bytes, MPI_BYTE,
                            origin_rank, OSHMPI_AM_PKT_ACK_TAG, OSHMPI_global.am_ack_comm_world));
}

/* Callback of IPUT operation. Receive data to local symm object.
 * No ACK is returned to origin PE. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_rma_am_iput_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt)
{
    void *dest = NULL;
    OSHMPI_am_iput_pkt_t *iput_pkt = &pkt->iput;

    OSHMPI_translate_disp_to_vaddr(iput_pkt->sobj_handle, iput_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    MPI_Datatype target_type = MPI_DATATYPE_NULL;
    size_t target_count = 0;
    OSHMPI_create_strided_dtype(iput_pkt->nelems, iput_pkt->target_st,
                                OSHMPI_global.am_datatypes_table[iput_pkt->mpi_type_idx],
                                0 /* no required extent */ ,
                                &target_count, &target_type);

    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Recv(dest, target_count, target_type, origin_rank,
                            OSHMPI_AM_PKT_DATA_TAG, OSHMPI_global.am_comm_world,
                            MPI_STATUS_IGNORE));

    OSHMPI_free_strided_dtype(OSHMPI_global.am_datatypes_table[iput_pkt->mpi_type_idx],
                              &target_type);

    if (target_type != OSHMPI_global.am_datatypes_table[iput_pkt->mpi_type_idx])
        OSHMPI_CALLMPI(MPI_Type_free(&target_type));
}

/* Callback of IGET operation. Send data from local symm object to origin PE. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_rma_am_iget_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt)
{
    OSHMPI_am_iget_pkt_t *iget_pkt = &pkt->iget;
    void *dest = NULL;

    OSHMPI_translate_disp_to_vaddr(iget_pkt->sobj_handle, iget_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    MPI_Datatype target_type = MPI_DATATYPE_NULL;
    size_t target_count = 0;
    OSHMPI_create_strided_dtype(iget_pkt->nelems, iget_pkt->target_st,
                                OSHMPI_global.am_datatypes_table[iget_pkt->mpi_type_idx],
                                0 /* no required extent */ ,
                                &target_count, &target_type);

    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Send(dest, target_count, target_type,
                            origin_rank, OSHMPI_AM_PKT_ACK_TAG, OSHMPI_global.am_ack_comm_world));

    OSHMPI_free_strided_dtype(OSHMPI_global.am_datatypes_table[iget_pkt->mpi_type_idx],
                              &target_type);
}

/* Issue a PUT operation. Return immediately after sent PUT packet (local complete) */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_rma_am_put(OSHMPI_ictx_t * ictx,
                                                   MPI_Datatype mpi_type, size_t typesz,
                                                   const void *origin_addr, MPI_Aint target_disp,
                                                   size_t nelems, int pe, uint32_t sobj_handle)
{
    OSHMPI_am_pkt_t pkt;
    OSHMPI_am_put_pkt_t *put_pkt = &pkt.put;

    pkt.type = OSHMPI_AM_PKT_PUT;
    put_pkt->target_disp = target_disp;
    put_pkt->bytes = typesz * nelems;
    put_pkt->sobj_handle = sobj_handle;

    OSHMPI_am_progress_mpi_send(&pkt, sizeof(OSHMPI_am_pkt_t), MPI_BYTE, pe, OSHMPI_AM_PKT_TAG,
                                OSHMPI_global.am_comm_world);

    OSHMPI_am_progress_mpi_send(origin_addr, nelems, mpi_type, pe, OSHMPI_AM_PKT_DATA_TAG,
                                OSHMPI_global.am_comm_world);
    OSHMPI_DBGMSG
        ("packet type %d, sobj_handle 0x%x, target %d, bytes %ld, disp 0x%lx\n",
         pkt.type, put_pkt->sobj_handle, pe, put_pkt->bytes, put_pkt->target_disp);

    /* Indicate outstanding AM */
    OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_global.am_outstanding_op_flags[pe], 1);
}

/* Issue a GET operation. Return after receiving return value. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_rma_am_get(OSHMPI_ictx_t * ictx,
                                                   MPI_Datatype mpi_type, size_t typesz,
                                                   void *origin_addr, MPI_Aint target_disp,
                                                   size_t nelems, int pe, uint32_t sobj_handle)
{
    OSHMPI_am_pkt_t pkt;
    OSHMPI_am_get_pkt_t *get_pkt = &pkt.get;

    pkt.type = OSHMPI_AM_PKT_GET;
    get_pkt->target_disp = target_disp;
    get_pkt->bytes = typesz * nelems;
    get_pkt->sobj_handle = sobj_handle;

    OSHMPI_am_progress_mpi_send(&pkt, sizeof(OSHMPI_am_pkt_t), MPI_BYTE, pe, OSHMPI_AM_PKT_TAG,
                                OSHMPI_global.am_comm_world);

    OSHMPI_am_progress_mpi_recv(origin_addr, nelems, mpi_type, pe, OSHMPI_AM_PKT_ACK_TAG,
                                OSHMPI_global.am_ack_comm_world, MPI_STATUS_IGNORE);

    OSHMPI_DBGMSG
        ("packet type %d, sobj_handle 0x%x, target %d, bytes %ld, disp 0x%lx\n",
         pkt.type, get_pkt->sobj_handle, pe, get_pkt->bytes, get_pkt->target_disp);

    /* Reset flag since remote PE should have finished previous put
     * before handling this get. */
    OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_global.am_outstanding_op_flags[pe], 0);
}

/* Issue a strided PUT operation. Return immediately after sent PUT packet (local complete) */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_rma_am_iput(OSHMPI_ictx_t * ictx,
                                                    MPI_Datatype mpi_type,
                                                    OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                    const void *origin_addr, MPI_Aint target_disp,
                                                    ptrdiff_t origin_st, ptrdiff_t target_st,
                                                    size_t nelems, int pe, uint32_t sobj_handle)
{
    OSHMPI_am_pkt_t pkt;
    OSHMPI_am_iput_pkt_t *iput_pkt = &pkt.iput;

    pkt.type = OSHMPI_AM_PKT_IPUT;
    iput_pkt->target_disp = target_disp;
    iput_pkt->mpi_type_idx = mpi_type_idx;
    iput_pkt->target_st = target_st;
    iput_pkt->nelems = nelems;
    iput_pkt->sobj_handle = sobj_handle;

    OSHMPI_am_progress_mpi_send(&pkt, sizeof(OSHMPI_am_pkt_t), MPI_BYTE, pe, OSHMPI_AM_PKT_TAG,
                                OSHMPI_global.am_comm_world);

    MPI_Datatype origin_type = MPI_DATATYPE_NULL;
    size_t origin_count = 0;
    OSHMPI_create_strided_dtype(nelems, origin_st, mpi_type, 0 /* no required extent */ ,
                                &origin_count, &origin_type);

    OSHMPI_am_progress_mpi_send(origin_addr, origin_count, origin_type, pe, OSHMPI_AM_PKT_DATA_TAG,
                                OSHMPI_global.am_comm_world);
    OSHMPI_DBGMSG("packet type %d, sobj_handle 0x%x, target %d, datatype idx %d, "
                  "origin_st 0x%lx, target_st 0x%lx, nelems %ld, disp 0x%lx\n",
                  pkt.type, iput_pkt->sobj_handle, pe, mpi_type_idx, origin_st, target_st, nelems,
                  iput_pkt->target_disp);

    /* Indicate outstanding AM */
    OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_global.am_outstanding_op_flags[pe], 1);
    OSHMPI_free_strided_dtype(mpi_type, &origin_type);
}

/* Issue a strided GET operation. Return after receiving return value. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_rma_am_iget(OSHMPI_ictx_t * ictx,
                                                    MPI_Datatype mpi_type,
                                                    OSHMPI_am_mpi_datatype_index_t mpi_type_idx,
                                                    void *origin_addr, MPI_Aint target_disp,
                                                    ptrdiff_t origin_st, ptrdiff_t target_st,
                                                    size_t nelems, int pe, uint32_t sobj_handle)
{
    OSHMPI_am_pkt_t pkt;
    OSHMPI_am_iget_pkt_t *iget_pkt = &pkt.iget;

    pkt.type = OSHMPI_AM_PKT_IGET;
    iget_pkt->target_disp = target_disp;
    iget_pkt->mpi_type_idx = mpi_type_idx;
    iget_pkt->target_st = target_st;
    iget_pkt->nelems = nelems;
    iget_pkt->sobj_handle = sobj_handle;

    OSHMPI_am_progress_mpi_send(&pkt, sizeof(OSHMPI_am_pkt_t), MPI_BYTE, pe, OSHMPI_AM_PKT_TAG,
                                OSHMPI_global.am_comm_world);

    MPI_Datatype origin_type = MPI_DATATYPE_NULL;
    size_t origin_count = 0;
    OSHMPI_create_strided_dtype(nelems, origin_st, mpi_type, 0 /* no required extent */ ,
                                &origin_count, &origin_type);

    OSHMPI_am_progress_mpi_recv(origin_addr, origin_count, origin_type, pe, OSHMPI_AM_PKT_ACK_TAG,
                                OSHMPI_global.am_ack_comm_world, MPI_STATUS_IGNORE);

    if (origin_type != mpi_type)
        OSHMPI_CALLMPI(MPI_Type_free(&origin_type));

    OSHMPI_DBGMSG("packet type %d, sobj_handle 0x%x, target %d, datatype idx %d, "
                  "origin_st 0x%lx, target_st 0x%lx, nelems %ld, disp 0x%lx\n",
                  pkt.type, iget_pkt->sobj_handle, pe, mpi_type_idx, origin_st, target_st, nelems,
                  iget_pkt->target_disp);

    /* Reset flag since remote PE should have finished previous put
     * before handling this get. */
    OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_global.am_outstanding_op_flags[pe], 0);
    OSHMPI_free_strided_dtype(mpi_type, &origin_type);
}

#endif /* INTERNAL_RMA_AM_IMPL_H */

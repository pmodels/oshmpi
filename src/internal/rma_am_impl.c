/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include "oshmpi_impl.h"

/* Callback of PUT operation. Receive data to local symm object.
 * No ACK is returned to origin PE. */
void OSHMPI_rma_am_put_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt)
{
    void *dest = NULL;
    OSHMPI_am_put_pkt_t *put_pkt = &pkt->put;

    OSHMPI_sobj_trans_disp_to_vaddr(put_pkt->sobj_handle, put_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Recv(dest, put_pkt->bytes, MPI_BYTE, origin_rank,
                            put_pkt->ptag, OSHMPI_am.comm, MPI_STATUS_IGNORE));
}

/* Callback of GET operation. Send data from local symm object to origin PE. */
void OSHMPI_rma_am_get_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt)
{
    OSHMPI_am_get_pkt_t *get_pkt = &pkt->get;
    void *dest = NULL;

    OSHMPI_sobj_trans_disp_to_vaddr(get_pkt->sobj_handle, get_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Send(dest, get_pkt->bytes, MPI_BYTE,
                            origin_rank, get_pkt->ptag, OSHMPI_am.ack_comm));
}

/* Callback of IPUT operation. Receive data to local symm object.
 * No ACK is returned to origin PE. */
void OSHMPI_rma_am_iput_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt)
{
    void *dest = NULL;
    OSHMPI_am_iput_pkt_t *iput_pkt = &pkt->iput;

    OSHMPI_sobj_trans_disp_to_vaddr(iput_pkt->sobj_handle, iput_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    MPI_Datatype target_type = MPI_DATATYPE_NULL;
    size_t target_count = 0;
    OSHMPI_create_strided_dtype(iput_pkt->nelems, iput_pkt->target_st,
                                OSHMPI_am.datatypes_table[iput_pkt->mpi_type_idx],
                                0 /* no required extent */ ,
                                &target_count, &target_type);

    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Recv(dest, target_count, target_type, origin_rank,
                            iput_pkt->ptag, OSHMPI_am.comm, MPI_STATUS_IGNORE));

    OSHMPI_free_strided_dtype(OSHMPI_am.datatypes_table[iput_pkt->mpi_type_idx], &target_type);
}

/* Callback of IGET operation. Send data from local symm object to origin PE. */
void OSHMPI_rma_am_iget_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt)
{
    OSHMPI_am_iget_pkt_t *iget_pkt = &pkt->iget;
    void *dest = NULL;

    OSHMPI_sobj_trans_disp_to_vaddr(iget_pkt->sobj_handle, iget_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    MPI_Datatype target_type = MPI_DATATYPE_NULL;
    size_t target_count = 0;
    OSHMPI_create_strided_dtype(iget_pkt->nelems, iget_pkt->target_st,
                                OSHMPI_am.datatypes_table[iget_pkt->mpi_type_idx],
                                0 /* no required extent */ ,
                                &target_count, &target_type);

    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Send(dest, target_count, target_type,
                            origin_rank, iget_pkt->ptag, OSHMPI_am.ack_comm));

    OSHMPI_free_strided_dtype(OSHMPI_am.datatypes_table[iget_pkt->mpi_type_idx], &target_type);
}

void OSHMPI_rma_am_initialize(void)
{
    OSHMPI_am_cb_regist(OSHMPI_AM_PKT_PUT, "PUT", OSHMPI_rma_am_put_pkt_cb);
    OSHMPI_am_cb_regist(OSHMPI_AM_PKT_GET, "GET", OSHMPI_rma_am_get_pkt_cb);
    OSHMPI_am_cb_regist(OSHMPI_AM_PKT_IPUT, "IPUT", OSHMPI_rma_am_iput_pkt_cb);
    OSHMPI_am_cb_regist(OSHMPI_AM_PKT_IGET, "IGET", OSHMPI_rma_am_iget_pkt_cb);
}

void OSHMPI_rma_am_finalize(void)
{

}

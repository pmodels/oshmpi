/* -*- C -*-
 *
 * Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
 * 
 * This file is part of the Portals SHMEM software package. For license
 * information, see the LICENSE file in the top level directory of the
 * distribution.
 *
 */

#ifndef TRANSPORT_MPI_H
#define TRANSPORT_MPI_H

#include <complex.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "shmem_free_list.h"

int shmem_transport_mpi3_init(long eager_size);

int shmem_transport_mpi3_startup(void);

int shmem_transport_mpi3_fini(void);

static inline
int
shmem_transport_mpi3_quiet(void)
{
    int ret;
    ptl_ct_event_t ct;

    /* wait for remote completion (acks) of all pending events */
    ret = PtlCTWait(shmem_transport_mpi3_put_ct_h, 
                    shmem_transport_mpi3_pending_put_counter, &ct);
    if (PTL_OK != ret) { return ret; }
    if (ct.failure != 0) { return -1; }

    return 0;
}


static inline
int
shmem_transport_mpi3_fence(void)
{
    return shmem_transport_mpi3_quiet();
}


static inline
void
shmem_transport_mpi3_put_small(void *target, const void *source, size_t len, int pe)
{
    int ret;
    ptl_process_t peer;
    ptl_pt_index_t pt;
    long offset;
    ptl_handle_md_t md_h;
    void *base;

    peer.rank = pe;
    mpi3_GET_REMOTE_ACCESS(target, pt, offset);

    assert(len <= shmem_transport_mpi3_max_volatile_size);

    shmem_transport_mpi3_get_md(source, shmem_transport_mpi3_put_volatile_md_h,
                                    &md_h, &base);

    ret = PtlPut(md_h,
                 (ptl_size_t) ((char*) source - (char*) base),
                 len,
                 PTL_CT_ACK_REQ,
                 peer,
                 pt,
                 0,
                 offset,
                 NULL,
                 0);
    if (PTL_OK != ret) { RAISE_ERROR(ret); }
    shmem_transport_mpi3_pending_put_counter++;
}


static inline
void
shmem_transport_mpi3_put_nb_internal(void *target, const void *source, size_t len,
                                int pe, long *completion, ptl_pt_index_t data_pt,
                                ptl_pt_index_t heap_pt)
{
    int ret;
    ptl_process_t peer;
    ptl_pt_index_t pt;
    long offset;
    ptl_handle_md_t md_h;
    void *base;

    peer.rank = pe;
#ifdef ENABLE_REMOTE_VIRTUAL_ADDRESSING
    mpi3_GET_REMOTE_ACCESS_ONEPT(target, pt, offset, data_pt);
#else
    mpi3_GET_REMOTE_ACCESS_TWOPT(target, pt, offset, data_pt, heap_pt);
#endif

    if (len <= shmem_transport_mpi3_max_volatile_size) {
        shmem_transport_mpi3_get_md(source, shmem_transport_mpi3_put_volatile_md_h,
                                        &md_h, &base);

        ret = PtlPut(md_h,
                     (ptl_size_t) ((char*) source - (char*) base),
                     len,
                     PTL_CT_ACK_REQ,
                     peer,
                     pt,
                     0,
                     offset,
                     NULL,
                     0);
        if (PTL_OK != ret) { RAISE_ERROR(ret); }

    }
    shmem_transport_mpi3_pending_put_counter++;
}


static inline
void
shmem_transport_mpi3_put_nb(void *target, const void *source, size_t len,
                                int pe, long *completion)
{
#ifdef ENABLE_REMOTE_VIRTUAL_ADDRESSING
    shmem_transport_mpi3_put_nb_internal(target, source, len, pe,
                                             completion,
                                             shmem_transport_mpi3_pt,
                                             -1);
#else
    shmem_transport_mpi3_put_nb_internal(target, source, len, pe,
                                             completion,
                                             shmem_transport_mpi3_data_pt,
                                             shmem_transport_mpi3_heap_pt);
#endif
}


static inline
void
shmem_transport_mpi3_put_wait(long *completion)
{
    while (*completion > 0) {
        shmem_transport_mpi3_drain_eq();
    }
}


static inline
void
shmem_transport_mpi3_get(void *target, const void *source, size_t len, int pe)
{
    int ret;
    ptl_process_t peer;
    ptl_pt_index_t pt;
    long offset;
    ptl_handle_md_t md_h;
    void *base;

    peer.rank = pe;
    mpi3_GET_REMOTE_ACCESS(source, pt, offset);

    shmem_transport_mpi3_get_md(target, shmem_transport_mpi3_get_md_h,
                                    &md_h, &base);

    ret = PtlGet(md_h,
                 (ptl_size_t) ((char*) target - (char*) base),
                 len,
                 peer,
                 pt,
                 0,
                 offset,
                 0);
    if (PTL_OK != ret) { RAISE_ERROR(ret); }
    shmem_transport_mpi3_pending_get_counter++;
}


static inline
void
shmem_transport_mpi3_get_wait(void)
{
    int ret;
    ptl_ct_event_t ct;

    ret = PtlCTWait(shmem_transport_mpi3_get_ct_h, 
                    shmem_transport_mpi3_pending_get_counter,
                    &ct);
    if (PTL_OK != ret) { RAISE_ERROR(ret); }
    if (ct.failure != 0) { RAISE_ERROR(ct.failure); }
}


static inline
void
shmem_transport_mpi3_swap(void *target, void *source, void *dest, size_t len, 
                              int pe, ptl_datatype_t datatype)
{
    int ret;
    ptl_process_t peer;
    ptl_pt_index_t pt;
    long offset;
    ptl_handle_md_t get_md_h;
    void *get_base;
    ptl_handle_md_t put_md_h;
    void *put_base;

    peer.rank = pe;
    mpi3_GET_REMOTE_ACCESS(target, pt, offset);

    assert(len <= sizeof(long double complex));
    assert(len <= shmem_transport_mpi3_max_volatile_size);

    shmem_transport_mpi3_get_md(dest, shmem_transport_mpi3_get_md_h,
                                    &get_md_h, &get_base);
    shmem_transport_mpi3_get_md(source, shmem_transport_mpi3_put_volatile_md_h,
                                    &put_md_h, &put_base);

    /* note: No ack is generated on the ct associated with the
       volatile md because the reply comes back on the get md.  So no
       need to increment the put counter */
    ret = PtlSwap(get_md_h,
                  (ptl_size_t) ((char*) dest - (char*) get_base),
                  put_md_h,
                  (ptl_size_t) ((char*) source - (char*) put_base),
                  len,
                  peer,
                  pt,
                  0,
                  offset,
                  NULL,
                  0,
                  NULL,
                  PTL_SWAP,
                  datatype);
    if (PTL_OK != ret) { RAISE_ERROR(ret); }
    shmem_transport_mpi3_pending_get_counter++;
}


static inline
void
shmem_transport_mpi3_cswap(void *target, void *source, void *dest, void *operand, size_t len, 
                               int pe, ptl_datatype_t datatype)
{
    int ret;
    ptl_process_t peer;
    ptl_pt_index_t pt;
    long offset;
    ptl_handle_md_t get_md_h;
    void *get_base;
    ptl_handle_md_t put_md_h;
    void *put_base;

    peer.rank = pe;
    mpi3_GET_REMOTE_ACCESS(target, pt, offset);

    assert(len <= sizeof(long double complex));
    assert(len <= shmem_transport_mpi3_max_volatile_size);

    shmem_transport_mpi3_get_md(dest, shmem_transport_mpi3_get_md_h,
                                    &get_md_h, &get_base);
    shmem_transport_mpi3_get_md(source, shmem_transport_mpi3_put_volatile_md_h,
                                    &put_md_h, &put_base);

    /* note: No ack is generated on the ct associated with the
       volatile md because the reply comes back on the get md.  So no
       need to increment the put counter */
    ret = PtlSwap(get_md_h,
                  (ptl_size_t) ((char*) dest - (char*) get_base),
                  put_md_h,
                  (ptl_size_t) ((char*) source - (char*) put_base),
                  len,
                  peer,
                  pt,
                  0,
                  offset,
                  NULL,
                  0,
                  operand,
                  PTL_CSWAP,
                  datatype);
    if (PTL_OK != ret) { RAISE_ERROR(ret); }
    shmem_transport_mpi3_pending_get_counter++;
}


static inline
void
shmem_transport_mpi3_mswap(void *target, void *source, void *dest, void *mask, size_t len, 
                               int pe, ptl_datatype_t datatype)
{
    int ret;
    ptl_process_t peer;
    ptl_pt_index_t pt;
    long offset;
    ptl_handle_md_t get_md_h;
    void *get_base;
    ptl_handle_md_t put_md_h;
    void *put_base;

    peer.rank = pe;
    mpi3_GET_REMOTE_ACCESS(target, pt, offset);

    assert(len <= sizeof(long double complex));
    assert(len <= shmem_transport_mpi3_max_volatile_size);

    shmem_transport_mpi3_get_md(dest, shmem_transport_mpi3_get_md_h,
                                    &get_md_h, &get_base);
    shmem_transport_mpi3_get_md(source, shmem_transport_mpi3_put_volatile_md_h,
                                    &put_md_h, &put_base);

    /* note: No ack is generated on the ct associated with the
       volatile md because the reply comes back on the get md.  So no
       need to increment the put counter */
    ret = PtlSwap(get_md_h,
                  (ptl_size_t) ((char*) dest - (char*) get_base),
                  put_md_h,
                  (ptl_size_t) ((char*) source - (char*) put_base),
                  len,
                  peer,
                  pt,
                  0,
                  offset,
                  NULL,
                  0,
                  mask,
                  PTL_MSWAP,
                  datatype);
    if (PTL_OK != ret) { RAISE_ERROR(ret); }
    shmem_transport_mpi3_pending_get_counter++;
}


static inline
void
shmem_transport_mpi3_atomic_small(void *target, void *source, size_t len,
                                       int pe, ptl_op_t op, ptl_datatype_t datatype)
{
    int ret;
    ptl_pt_index_t pt;
    long offset;
    ptl_process_t peer;
    ptl_handle_md_t md_h;
    void *base;

    peer.rank = pe;
    mpi3_GET_REMOTE_ACCESS(target, pt, offset);

    assert(len <= shmem_transport_mpi3_max_volatile_size);

    shmem_transport_mpi3_get_md(source, shmem_transport_mpi3_put_volatile_md_h,
                                    &md_h, &base);

    ret = PtlAtomic(md_h,
                    (ptl_size_t) ((char*) source - (char*) base),
                    len,
                    PTL_CT_ACK_REQ,
                    peer,
                    pt,
                    0,
                    offset,
                    NULL,
                    0,
                    op,
                    datatype);
    if (PTL_OK != ret) { RAISE_ERROR(ret); }
    shmem_transport_mpi3_pending_put_counter += 1;
}


static inline
void
shmem_transport_mpi3_atomic_nb(void *target, void *source, size_t len,
                                   int pe, ptl_op_t op, ptl_datatype_t datatype,
                                   long *completion)
{
    int ret;
    ptl_pt_index_t pt;
    long offset;
    ptl_process_t peer;
    ptl_handle_md_t md_h;
    void *base;

    peer.rank = pe;
    mpi3_GET_REMOTE_ACCESS(target, pt, offset);

    if (len <= shmem_transport_mpi3_max_volatile_size) {
        shmem_transport_mpi3_get_md(source, shmem_transport_mpi3_put_volatile_md_h,
                                        &md_h, &base);

        ret = PtlAtomic(md_h,
                        (ptl_size_t) ((char*) source - (char*) base),
                        len,
                        PTL_CT_ACK_REQ,
                        peer,
                        pt,
                        0,
                        offset,
                        NULL,
                        0,
                        op,
                        datatype);
        if (PTL_OK != ret) { RAISE_ERROR(ret); }
        shmem_transport_mpi3_pending_put_counter++;

    }
}


static inline
void
shmem_transport_mpi3_fetch_atomic(void *target, void *source, void *dest, size_t len,
                                      int pe, ptl_op_t op, ptl_datatype_t datatype)
{
    int ret;
    ptl_pt_index_t pt;
    long offset;
    ptl_process_t peer;
    ptl_handle_md_t get_md_h;
    void *get_base;
    ptl_handle_md_t put_md_h;
    void *put_base;

    peer.rank = pe;
    mpi3_GET_REMOTE_ACCESS(target, pt, offset);

    assert(len <= shmem_transport_mpi3_max_fetch_atomic_size);
    assert(len <= shmem_transport_mpi3_max_volatile_size);

    shmem_transport_mpi3_get_md(source, shmem_transport_mpi3_get_md_h,
                                    &get_md_h, &get_base);
    shmem_transport_mpi3_get_md(source, shmem_transport_mpi3_put_volatile_md_h,
                                    &put_md_h, &put_base);

    /* note: No ack is generated on the ct associated with the
       volatile md because the reply comes back on the get md.  So no
       need to increment the put counter */
    ret = PtlFetchAtomic(get_md_h,
                         (ptl_size_t) ((char*) dest - (char*) get_base),
                         put_md_h,
                         (ptl_size_t) ((char*) source - (char*) put_base),
                         len,
                         peer,
                         pt,
                         0,
                         offset,
                         NULL,
                         0,
                         op,
                         datatype);
    if (PTL_OK != ret) { RAISE_ERROR(ret); }
    shmem_transport_mpi3_pending_get_counter++;
}


#endif

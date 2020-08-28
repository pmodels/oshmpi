/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#ifndef INTERNAL_AMO_AM_IMPL_H
#define INTERNAL_AMO_AM_IMPL_H

#include "oshmpi_impl.h"

#define OSHMPI_AMO_OP_FP_IMPL(mpi_op_idx, c_type, a_ptr, b_ptr)              \
        switch(mpi_op_idx) {                                                            \
            case OSHMPI_AM_MPI_NO_OP:                                                  \
                break;                                                                  \
            case OSHMPI_AM_MPI_REPLACE:                                                \
                *(c_type *) (b_ptr) = *(c_type *) (a_ptr);                              \
                break;                                                                  \
            case OSHMPI_AM_MPI_SUM:                                                    \
                *(c_type *) (b_ptr) += *(c_type *) (a_ptr);                             \
                break;                                                                  \
            default:                                                                    \
                OSHMPI_ERR_ABORT("Unsupported MPI op index for floating point: %d\n", (int) mpi_op_idx);   \
                break;                                                                                     \
        }

#define OSHMPI_AMO_OP_INT_IMPL(mpi_op_idx, c_type, a_ptr, b_ptr)              \
        switch(mpi_op_idx) {                                                            \
            case OSHMPI_AM_MPI_BAND:                                                   \
                *(c_type *) (b_ptr) = (*(c_type *) (a_ptr)) & (*(c_type *) (b_ptr));    \
                break;                                                                  \
            case OSHMPI_AM_MPI_BOR:                                                    \
                *(c_type *) (b_ptr) = (*(c_type *) (a_ptr)) | (*(c_type *) (b_ptr));    \
                break;                                                                  \
            case OSHMPI_AM_MPI_BXOR:                                                   \
                *(c_type *) (b_ptr) = (*(c_type *) (a_ptr)) ^ (*(c_type *) (b_ptr));    \
                break;                                                                  \
            case OSHMPI_AM_MPI_NO_OP:                                                  \
                break;                                                                  \
            case OSHMPI_AM_MPI_REPLACE:                                                \
                *(c_type *) (b_ptr) = *(c_type *) (a_ptr);                              \
                break;                                                                  \
            case OSHMPI_AM_MPI_SUM:                                                    \
                *(c_type *) (b_ptr) += *(c_type *) (a_ptr);                             \
                break;                                                                  \
            default:                                                                    \
                OSHMPI_ERR_ABORT("Unsupported MPI op index for integer: %d\n", (int) mpi_op_idx);   \
                break;                                                                              \
        }

#define OSHMPI_AMO_OP_TYPE_IMPL(mpi_type_idx) do {              \
    switch(mpi_type_idx) {                                      \
        case OSHMPI_AM_MPI_INT:                                \
           OSHMPI_OP_INT_MACRO(int); break;                     \
        case OSHMPI_AM_MPI_LONG:                               \
           OSHMPI_OP_INT_MACRO(long); break;                    \
        case OSHMPI_AM_MPI_LONG_LONG:                          \
           OSHMPI_OP_INT_MACRO(long long); break;               \
        case OSHMPI_AM_MPI_UNSIGNED:                           \
           OSHMPI_OP_INT_MACRO(unsigned int); break;            \
        case OSHMPI_AM_MPI_UNSIGNED_LONG:                      \
           OSHMPI_OP_INT_MACRO(unsigned long); break;           \
        case OSHMPI_AM_MPI_UNSIGNED_LONG_LONG:                 \
           OSHMPI_OP_INT_MACRO(unsigned long long); break;      \
        case OSHMPI_AM_MPI_INT32_T:                            \
           OSHMPI_OP_INT_MACRO(int32_t); break;                 \
        case OSHMPI_AM_MPI_INT64_T:                            \
           OSHMPI_OP_INT_MACRO(int64_t); break;                 \
        case OSHMPI_AM_OSHMPI_MPI_SIZE_T:                      \
           OSHMPI_OP_INT_MACRO(size_t); break;                  \
        case OSHMPI_AM_MPI_UINT32_T:                           \
           OSHMPI_OP_INT_MACRO(uint32_t); break;                \
        case OSHMPI_AM_MPI_UINT64_T:                           \
           OSHMPI_OP_INT_MACRO(uint64_t); break;                \
        case OSHMPI_AM_OSHMPI_MPI_PTRDIFF_T:                   \
           OSHMPI_OP_INT_MACRO(ptrdiff_t); break;               \
        case OSHMPI_AM_MPI_FLOAT:                              \
           OSHMPI_OP_FP_MACRO(float); break;                    \
        case OSHMPI_AM_MPI_DOUBLE:                             \
           OSHMPI_OP_FP_MACRO(double); break;                   \
        default:                                                \
            OSHMPI_ERR_ABORT("Unsupported MPI type index: %d\n", mpi_type_idx);   \
            break;                                              \
    }                                                           \
} while (0)

/* Callback of compare_and_swap AMO operation. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_cswap_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt)
{
    OSHMPI_am_datatype_t oldval;
    OSHMPI_am_cswap_pkt_t *cswap_pkt = &pkt->cswap;
    void *dest = NULL;
    void *oldval_ptr = &oldval, *cond_ptr = &cswap_pkt->cond, *value_ptr = &cswap_pkt->value;

    OSHMPI_sobj_trans_disp_to_vaddr(cswap_pkt->sobj_handle, cswap_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    /* Compute.
     * All AMOs are handled as active message, no lock needed.
     * We use same macro for floating point and integer types. */
#undef OSHMPI_OP_INT_MACRO
#undef OSHMPI_OP_FP_MACRO
#define OSHMPI_OP_INT_MACRO(c_type) do {           \
        *(c_type *) oldval_ptr = *(c_type *) dest;    \
        if (*(c_type *) dest == *(c_type *) cond_ptr) \
            *(c_type *) dest = *(c_type *) value_ptr; \
    } while (0)
#define OSHMPI_OP_FP_MACRO OSHMPI_OP_INT_MACRO
    OSHMPI_AMO_OP_TYPE_IMPL(cswap_pkt->mpi_type_idx);
#undef OSHMPI_OP_INT_MACRO
#undef OSHMPI_OP_FP_MACRO

    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Send(&oldval, 1, OSHMPI_global.am_datatypes_table[cswap_pkt->mpi_type_idx],
                            origin_rank, cswap_pkt->ptag, OSHMPI_global.am_ack_comm_world));
}

/* Callback of fetch (with op) AMO operation. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_fetch_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt)
{
    OSHMPI_am_datatype_t oldval;
    OSHMPI_am_fetch_pkt_t *fetch_pkt = &pkt->fetch;
    void *dest = NULL;
    void *oldval_ptr = &oldval, *value_ptr = &fetch_pkt->value;

    OSHMPI_sobj_trans_disp_to_vaddr(fetch_pkt->sobj_handle, fetch_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    /* Compute.
     * All AMOs are handled as active message, no lock needed.
     * We use different op set for floating point and integer types. */
#undef OSHMPI_OP_INT_MACRO
#undef OSHMPI_OP_FP_MACRO
#define OSHMPI_OP_INT_MACRO(c_type) do {                          \
        *(c_type *) oldval_ptr = *(c_type *) dest;                \
        OSHMPI_AMO_OP_INT_IMPL(fetch_pkt->mpi_op_idx, c_type, value_ptr, dest); \
    } while (0)
#define OSHMPI_OP_FP_MACRO(c_type) do {                          \
        *(c_type *) oldval_ptr = *(c_type *) dest;               \
        OSHMPI_AMO_OP_FP_IMPL(fetch_pkt->mpi_op_idx, c_type, value_ptr, dest); \
    } while (0)

    OSHMPI_AMO_OP_TYPE_IMPL(fetch_pkt->mpi_type_idx);

#undef OSHMPI_OP_INT_MACRO
#undef OSHMPI_OP_FP_MACRO

    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Send(&oldval, 1, OSHMPI_global.am_datatypes_table[fetch_pkt->mpi_type_idx],
                            origin_rank, fetch_pkt->ptag, OSHMPI_global.am_ack_comm_world));
}

/* Callback of post AMO operation. No ACK is returned to origin PE. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_am_post_pkt_cb(int origin_rank, OSHMPI_am_pkt_t * pkt)
{
    void *dest = NULL;
    OSHMPI_am_post_pkt_t *post_pkt = &pkt->post;
    void *value_ptr = &post_pkt->value;

    OSHMPI_sobj_trans_disp_to_vaddr(post_pkt->sobj_handle, post_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    /* Compute.
     * All AMOs are handled as active message, no lock needed.
     * We use different op set for floating point and integer types. */
#undef OSHMPI_OP_INT_MACRO
#undef OSHMPI_OP_FP_MACRO

#define OSHMPI_OP_INT_MACRO(c_type) do {                          \
        OSHMPI_AMO_OP_INT_IMPL(post_pkt->mpi_op_idx, c_type, value_ptr, dest); \
    } while (0)
#define OSHMPI_OP_FP_MACRO(c_type) do {                          \
        OSHMPI_AMO_OP_FP_IMPL(post_pkt->mpi_op_idx, c_type, value_ptr, dest); \
    } while (0)

    OSHMPI_AMO_OP_TYPE_IMPL(post_pkt->mpi_type_idx);

#undef OSHMPI_OP_INT_MACRO
#undef OSHMPI_OP_FP_MACRO
}

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
                                OSHMPI_global.am_comm_world);

    OSHMPI_am_progress_mpi_recv(oldval_ptr, 1, mpi_type, pe, cswap_pkt->ptag,
                                OSHMPI_global.am_ack_comm_world, MPI_STATUS_IGNORE);

    OSHMPI_DBGMSG
        ("packet type %d, sobj_handle 0x%x, target %d, datatype idx %d, addr %p, disp 0x%lx, ptag %d\n",
         pkt.type, cswap_pkt->sobj_handle, pe, mpi_type_idx, dest, cswap_pkt->target_disp,
         cswap_pkt->ptag);

    /* Reset flag since remote PE should have finished previous post
     * before handling this fetch. */
    OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_global.am_outstanding_op_flags[pe], 0);
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
                                OSHMPI_global.am_comm_world);

    OSHMPI_am_progress_mpi_recv(oldval_ptr, 1, mpi_type, pe, fetch_pkt->ptag,
                                OSHMPI_global.am_ack_comm_world, MPI_STATUS_IGNORE);

    OSHMPI_DBGMSG
        ("packet type %d, sobj_handle 0x%x, target %d, datatype idx %d, op idx %d, addr %p, disp 0x%lx, ptag %d\n",
         pkt.type, fetch_pkt->sobj_handle, pe, mpi_type_idx, op_idx, dest, fetch_pkt->target_disp,
         fetch_pkt->ptag);

    /* Reset flag since remote PE should have finished previous post
     * before handling this fetch. */
    OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_global.am_outstanding_op_flags[pe], 0);
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
                                OSHMPI_global.am_comm_world);
    OSHMPI_DBGMSG
        ("packet type %d, sobj_handle 0x%x, target %d, datatype idx %d, op idx %d, addr %p, disp 0x%lx\n",
         pkt.type, post_pkt->sobj_handle, pe, mpi_type_idx, op_idx, dest, post_pkt->target_disp);

    /* Indicate outstanding AMO */
    OSHMPI_ATOMIC_FLAG_STORE(OSHMPI_global.am_outstanding_op_flags[pe], 1);
}

#endif /* INTERNAL_AMO_AM_IMPL_H */

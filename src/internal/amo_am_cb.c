/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include "oshmpi_impl.h"

#define AMO_OP_TYPE_IMPL(mpi_type_idx) do {              \
    switch(mpi_type_idx) {                               \
        case OSHMPI_AMO_MPI_INT:                         \
           OP_INT_MACRO(int); break;                     \
        case OSHMPI_AMO_MPI_LONG:                        \
           OP_INT_MACRO(long); break;                    \
        case OSHMPI_AMO_MPI_LONG_LONG:                   \
           OP_INT_MACRO(long long); break;               \
        case OSHMPI_AMO_MPI_UNSIGNED:                    \
           OP_INT_MACRO(unsigned int); break;            \
        case OSHMPI_AMO_MPI_UNSIGNED_LONG:               \
           OP_INT_MACRO(unsigned long); break;           \
        case OSHMPI_AMO_MPI_UNSIGNED_LONG_LONG:          \
           OP_INT_MACRO(unsigned long long); break;      \
        case OSHMPI_AMO_MPI_INT32_T:                     \
           OP_INT_MACRO(int32_t); break;                 \
        case OSHMPI_AMO_MPI_INT64_T:                     \
           OP_INT_MACRO(int64_t); break;                 \
        case OSHMPI_AMO_OSHMPI_MPI_SIZE_T:               \
           OP_INT_MACRO(size_t); break;                  \
        case OSHMPI_AMO_MPI_UINT32_T:                    \
           OP_INT_MACRO(uint32_t); break;                \
        case OSHMPI_AMO_MPI_UINT64_T:                    \
           OP_INT_MACRO(uint64_t); break;                \
        case OSHMPI_AMO_OSHMPI_MPI_PTRDIFF_T:            \
           OP_INT_MACRO(ptrdiff_t); break;               \
        case OSHMPI_AMO_MPI_FLOAT:                       \
           OP_FP_MACRO(float); break;                    \
        case OSHMPI_AMO_MPI_DOUBLE:                      \
           OP_FP_MACRO(double); break;                   \
        default:                                         \
            OSHMPI_ERR_ABORT("Unsupported MPI type index: %d\n", mpi_type_idx);   \
            break;                                       \
    }                                                    \
} while (0)

#define AMO_OP_FP_IMPL(mpi_op_idx, c_type, a_ptr, b_ptr)              \
        switch(mpi_op_idx) {                                          \
            case OSHMPI_AMO_MPI_NO_OP:                                \
                break;                                                \
            case OSHMPI_AMO_MPI_REPLACE:                              \
                *(c_type *) (b_ptr) = *(c_type *) (a_ptr);            \
                break;                                                \
            case OSHMPI_AMO_MPI_SUM:                                  \
                *(c_type *) (b_ptr) += *(c_type *) (a_ptr);           \
                break;                                                \
            default:                                                  \
                OSHMPI_ERR_ABORT("Unsupported MPI op index for floating point: %d\n", (int) mpi_op_idx);   \
                break;                                                                                     \
        }

#define AMO_OP_INT_IMPL(mpi_op_idx, c_type, a_ptr, b_ptr)                               \
        switch(mpi_op_idx) {                                                            \
            case OSHMPI_AMO_MPI_BAND:                                                   \
                *(c_type *) (b_ptr) = (*(c_type *) (a_ptr)) & (*(c_type *) (b_ptr));    \
                break;                                                                  \
            case OSHMPI_AMO_MPI_BOR:                                                    \
                *(c_type *) (b_ptr) = (*(c_type *) (a_ptr)) | (*(c_type *) (b_ptr));    \
                break;                                                                  \
            case OSHMPI_AMO_MPI_BXOR:                                                   \
                *(c_type *) (b_ptr) = (*(c_type *) (a_ptr)) ^ (*(c_type *) (b_ptr));    \
                break;                                                                  \
            case OSHMPI_AMO_MPI_NO_OP:                                                  \
                break;                                                                  \
            case OSHMPI_AMO_MPI_REPLACE:                                                \
                *(c_type *) (b_ptr) = *(c_type *) (a_ptr);                              \
                break;                                                                  \
            case OSHMPI_AMO_MPI_SUM:                                                    \
                *(c_type *) (b_ptr) += *(c_type *) (a_ptr);                             \
                break;                                                                  \
            default:                                                                    \
                OSHMPI_ERR_ABORT("Unsupported MPI op index for integer: %d\n", (int) mpi_op_idx);   \
                break;                                                                              \
        }

/* Callback of compare_and_swap AMO operation. */
void OSHMPI_amo_cswap_pkt_cb(int origin_rank, OSHMPI_pkt_t * pkt)
{
    OSHMPI_amo_datatype_t oldval;
    OSHMPI_amo_cswap_pkt_t *cswap_pkt = &pkt->cswap;
    void *dest = NULL;
    void *oldval_ptr = &oldval, *cond_ptr = &cswap_pkt->cond, *value_ptr = &cswap_pkt->value;

    OSHMPI_translate_disp_to_vaddr(cswap_pkt->symm_obj_type, cswap_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    /* Compute.
     * All AMOs are handled as active message, no lock needed.
     * We use same macro for floating point and integer types. */
#undef OP_INT_MACRO
#undef OP_FP_MACRO
#define OP_INT_MACRO(c_type) do {           \
        *(c_type *) oldval_ptr = *(c_type *) dest;    \
        if (*(c_type *) dest == *(c_type *) cond_ptr) \
            *(c_type *) dest = *(c_type *) value_ptr; \
    } while (0)
#define OP_FP_MACRO OP_INT_MACRO
    AMO_OP_TYPE_IMPL(cswap_pkt->mpi_type_idx);
#undef OP_INT_MACRO
#undef OP_FP_MACRO

    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Send(&oldval, 1, OSHMPI_global.amo_datatypes_table[cswap_pkt->mpi_type_idx],
                            origin_rank, OSHMPI_PKT_AMO_ACK_TAG, OSHMPI_global.amo_ack_comm_world));
}

/* Callback of fetch (with op) AMO operation. */
void OSHMPI_amo_fetch_pkt_cb(int origin_rank, OSHMPI_pkt_t * pkt)
{
    OSHMPI_amo_datatype_t oldval;
    OSHMPI_amo_fetch_pkt_t *fetch_pkt = &pkt->fetch;
    void *dest = NULL;
    void *oldval_ptr = &oldval, *value_ptr = &fetch_pkt->value;

    OSHMPI_translate_disp_to_vaddr(fetch_pkt->symm_obj_type, fetch_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    /* Compute.
     * All AMOs are handled as active message, no lock needed.
     * We use different op set for floating point and integer types. */
#undef OP_INT_MACRO
#undef OP_FP_MACRO
#define OP_INT_MACRO(c_type) do {                          \
        *(c_type *) oldval_ptr = *(c_type *) dest;                \
        AMO_OP_INT_IMPL(fetch_pkt->mpi_op_idx, c_type, value_ptr, dest); \
    } while (0)
#define OP_FP_MACRO(c_type) do {                          \
        *(c_type *) oldval_ptr = *(c_type *) dest;               \
        AMO_OP_FP_IMPL(fetch_pkt->mpi_op_idx, c_type, value_ptr, dest); \
    } while (0)

    AMO_OP_TYPE_IMPL(fetch_pkt->mpi_type_idx);

#undef OP_INT_MACRO
#undef OP_FP_MACRO

    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Send(&oldval, 1, OSHMPI_global.amo_datatypes_table[fetch_pkt->mpi_type_idx],
                            origin_rank, OSHMPI_PKT_AMO_ACK_TAG, OSHMPI_global.amo_ack_comm_world));
}

/* Callback of post AMO operation. No ACK is returned to origin PE. */
void OSHMPI_amo_post_pkt_cb(int origin_rank, OSHMPI_pkt_t * pkt)
{
    void *dest = NULL;
    OSHMPI_amo_post_pkt_t *post_pkt = &pkt->post;
    void *value_ptr = &post_pkt->value;

    OSHMPI_translate_disp_to_vaddr(post_pkt->symm_obj_type, post_pkt->target_disp, &dest);
    OSHMPI_ASSERT(dest);

    /* Compute.
     * All AMOs are handled as active message, no lock needed.
     * We use different op set for floating point and integer types. */
#undef OP_INT_MACRO
#undef OP_FP_MACRO

#define OP_INT_MACRO(c_type) do {                          \
        AMO_OP_INT_IMPL(post_pkt->mpi_op_idx, c_type, value_ptr, dest); \
    } while (0)
#define OP_FP_MACRO(c_type) do {                          \
        AMO_OP_FP_IMPL(post_pkt->mpi_op_idx, c_type, value_ptr, dest); \
    } while (0)

    AMO_OP_TYPE_IMPL(post_pkt->mpi_type_idx);

#undef OP_INT_MACRO
#undef OP_FP_MACRO
}

/* Callback of flush synchronization. */
void OSHMPI_amo_flush_pkt_cb(int origin_rank, OSHMPI_pkt_t * pkt)
{
    /* Do not make AM progress in callback to avoid re-entry of progress loop. */
    OSHMPI_CALLMPI(MPI_Send(NULL, 0, MPI_BYTE, origin_rank, OSHMPI_PKT_AMO_ACK_TAG,
                            OSHMPI_global.amo_ack_comm_world));
}

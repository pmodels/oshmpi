/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef OSHMPI_AM_PKT_H
#define OSHMPI_AM_PKT_H

/* Ensure packet header variables can fit all possible types */
typedef union {
    int i_v;
    long l_v;
    long long ll_v;
    unsigned int ui_v;
    unsigned long ul_v;
    unsigned long long ull_v;
    int32_t i32_v;
    int64_t i64_v;
    uint32_t ui32_v;
    uint64_t ui64_v;
    size_t sz_v;
    ptrdiff_t ptr_v;
    float f_v;
    double d_v;
} OSHMPI_amo_datatype_t;

/* Define active message packet header. Note that we do not define
 * packet for ACK of each packet, because all routines that require ACK
 * are blocking, thus we can directly receive ACK without going through
 * the callback progress handling. */
typedef enum {
    OSHMPI_PKT_AMO_CSWAP,
    OSHMPI_PKT_AMO_FETCH,
    OSHMPI_PKT_AMO_POST,
    OSHMPI_PKT_AMO_FLUSH,
    OSHMPI_PKT_TERMINATE,       /* terminate async thread or consume the last AM recv */
    OSHMPI_PKT_MAX,
} OSHMPI_pkt_type_t;

typedef struct OSHMPI_amo_cswap_pkt {
    OSHMPI_amo_datatype_t cond;
    OSHMPI_amo_datatype_t value;
    OSHMPI_symm_obj_type_t symm_obj_type;
    OSHMPI_amo_mpi_datatype_index_t mpi_type_idx;
    int target_disp;
    size_t bytes;
} OSHMPI_amo_cswap_pkt_t;

typedef struct OSHMPI_amo_fetch_pkt {
    OSHMPI_amo_datatype_t value;
    OSHMPI_symm_obj_type_t symm_obj_type;
    OSHMPI_amo_mpi_datatype_index_t mpi_type_idx;
    OSHMPI_amo_mpi_op_index_t mpi_op_idx;
    int target_disp;
    size_t bytes;
} OSHMPI_amo_fetch_pkt_t;

typedef OSHMPI_amo_fetch_pkt_t OSHMPI_amo_post_pkt_t;
typedef struct {
} OSHMPI_amo_flush_pkt_t;

typedef struct OSHMPI_pkt {
    int type;
    union {
        OSHMPI_amo_cswap_pkt_t cswap;
        OSHMPI_amo_fetch_pkt_t fetch;
        OSHMPI_amo_post_pkt_t post;
        OSHMPI_amo_flush_pkt_t flush;
    };
} OSHMPI_pkt_t;

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_cswap_pkt_cb(int origin_rank, OSHMPI_pkt_t * pkt);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_fetch_pkt_cb(int origin_rank, OSHMPI_pkt_t * pkt);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_post_pkt_cb(int origin_rank, OSHMPI_pkt_t * pkt);
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_amo_flush_pkt_cb(int origin_rank, OSHMPI_pkt_t * pkt);
#endif /* OSHMPI_AM_PKT_H */

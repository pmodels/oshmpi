/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#ifndef INTERNAL_P2P_IMPL_H
#define INTERNAL_P2P_IMPL_H

#include "oshmpi_impl.h"

/* Macro compares two variables. Set ret to 1 if evaluates to true, otherwise 0.*/
#define OSHMPI_COMP(a, comp_op, b, ret /* OUT */) do {                             \
    switch (comp_op) {                                                             \
        case SHMEM_CMP_EQ:                                                         \
            ret = (a == b) ? 1 : 0; break;                                         \
        case SHMEM_CMP_NE:                                                         \
            ret = (a != b) ? 1 : 0; break;                                         \
        case SHMEM_CMP_GT:                                                         \
            ret = (a > b) ? 1 : 0; break;                                          \
        case SHMEM_CMP_GE:                                                         \
            ret = (a >= b) ? 1 : 0; break;                                         \
        case SHMEM_CMP_LT:                                                         \
            ret = (a < b) ? 1 : 0; break;                                          \
        case SHMEM_CMP_LE:                                                         \
            ret = (a <= b) ? 1 : 0; break;                                         \
        default:                                                                   \
            OSHMPI_ASSERT(comp_op < SHMEM_CMP_EQ || comp_op > SHMEM_CMP_LE);       \
            break;                                                                 \
    }                                                                              \
} while (0)

/* Compares two variables with specified operation, blocking wait until comparison evaluates to true. */
#define OSHMPI_WAIT_UNTIL(ivar, comp_op, comp_value, C_TYPE, MPI_TYPE) do {                         \
    C_TYPE tmp_var;                                                                                 \
    MPI_Aint target_disp = -1;                                                                      \
    OSHMPI_ictx_t *ictx = NULL;                                                                     \
    OSHMPI_sobj_attr_t *sobj_attr = NULL;                                                           \
    unsigned int comp_ret = 0;                                                                      \
                                                                                                    \
    OSHMPI_sobj_query_attr_ictx(SHMEM_CTX_DEFAULT, (const void *) ivar,                             \
                                OSHMPI_global.team_world_my_pe, &sobj_attr, &ictx);                                                 \
    OSHMPI_ASSERT(sobj_attr && ictx);                                                               \
    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, ivar, OSHMPI_global.team_world_my_pe,                \
                                    OSHMPI_ICTX_DISP_MODE(ictx), &target_disp);                     \
    OSHMPI_ASSERT(target_disp >= 0);                                                                \
    while (1) {                                                                                     \
        OSHMPI_CALLMPI(MPI_Fetch_and_op(NULL, &tmp_var, MPI_TYPE,                                   \
                                        OSHMPI_global.team_world_my_pe, target_disp, MPI_NO_OP,     \
                                        ictx->win));                                                \
        OSHMPI_CALLMPI(MPI_Win_flush_local(OSHMPI_global.team_world_my_pe, ictx->win));             \
        OSHMPI_COMP(tmp_var, comp_op, comp_value, comp_ret);                                        \
        if (comp_ret) break; /* skip AM progress if complete immediately */                         \
        OSHMPI_am_cb_progress();                                                                   \
        OSHMPI_progress_poll_mpi();                                                                 \
    }                                                                                               \
} while (0)

/* Nonblocking routine scompares two variables with specified operation.
 * Set test_ret to 1 if evaluates to true, otherwise 0.*/
#define OSHMPI_TEST(ivar, comp_op, comp_value, C_TYPE, MPI_TYPE, test_ret /* OUT */) do {       \
    C_TYPE tmp_var;                                                                             \
    MPI_Aint target_disp = -1;                                                                  \
    OSHMPI_ictx_t *ictx = NULL;                                                                 \
    OSHMPI_sobj_attr_t *sobj_attr = NULL;                                                       \
                                                                                                \
    OSHMPI_sobj_query_attr_ictx(SHMEM_CTX_DEFAULT, (const void *) ivar,                             \
                                OSHMPI_global.team_world_my_pe, &sobj_attr, &ictx);                 \
    OSHMPI_ASSERT(sobj_attr && ictx);                                                               \
    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, ivar, OSHMPI_global.team_world_my_pe,                \
                                    OSHMPI_ICTX_DISP_MODE(ictx), &target_disp);                     \
    OSHMPI_ASSERT(target_disp >= 0);                                                                \
    OSHMPI_CALLMPI(MPI_Fetch_and_op(NULL, &tmp_var, MPI_TYPE,                                       \
                                    OSHMPI_global.team_world_my_pe, target_disp, MPI_NO_OP,         \
                                    ictx->win));                                                    \
    OSHMPI_CALLMPI(MPI_Win_flush_local(OSHMPI_global.team_world_my_pe, ictx->win));                 \
    OSHMPI_COMP(tmp_var, comp_op, comp_value, test_ret);                                       \
    if (!test_ret) /* Skip progress if complete immediately */                                 \
        OSHMPI_am_cb_progress();                                                               \
} while (0)

#endif /* INTERNAL_P2P_IMPL_H */

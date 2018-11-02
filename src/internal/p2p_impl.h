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
    MPI_Win win = MPI_WIN_NULL;                                                                     \
    unsigned int comp_ret = 0;                                                                      \
                                                                                                    \
    OSHMPI_translate_win_and_disp(ivar, &win, &target_disp);                                        \
    OSHMPI_ASSERT(target_disp >= 0 && win != MPI_WIN_NULL);                                         \
    while (1) {                                                                                     \
        OSHMPI_CALLMPI(MPI_Fetch_and_op(NULL, &tmp_var, MPI_TYPE,                                   \
                                        OSHMPI_global.world_rank, target_disp, MPI_NO_OP, win));    \
        OSHMPI_CALLMPI(MPI_Win_flush_local(OSHMPI_global.world_rank, win));                         \
        OSHMPI_COMP(tmp_var, comp_op, comp_value, comp_ret);                                        \
        if (comp_ret) break; /* skip AM progress if complete immediately */                         \
        OSHMPI_amo_cb_progress();                                                                   \
    }                                                                                               \
} while (0)

/* Nonblocking routine scompares two variables with specified operation.
 * Set test_ret to 1 if evaluates to true, otherwise 0.*/
#define OSHMPI_TEST(ivar, comp_op, comp_value, C_TYPE, MPI_TYPE, test_ret /* OUT */) do {       \
    C_TYPE tmp_var;                                                                             \
    MPI_Aint target_disp = -1;                                                                  \
    MPI_Win win = MPI_WIN_NULL;                                                                 \
                                                                                                \
    OSHMPI_translate_win_and_disp(ivar, &win, &target_disp);                                    \
    OSHMPI_ASSERT(target_disp >= 0 && win != MPI_WIN_NULL);                                     \
    OSHMPI_CALLMPI(MPI_Fetch_and_op(NULL, &tmp_var, MPI_TYPE,                                   \
                                    OSHMPI_global.world_rank, target_disp, MPI_NO_OP, win));    \
    OSHMPI_CALLMPI(MPI_Win_flush_local(OSHMPI_global.world_rank, win));                         \
    OSHMPI_COMP(tmp_var, comp_op, comp_value, test_ret);                                        \
    if (!test_ret) /* Skip progress if complete immediately */                                  \
        OSHMPI_amo_cb_progress();                                                               \
} while (0)

#endif /* INTERNAL_P2P_IMPL_H */

/* -*- C -*-
 *
 * Copyright 2013 Argonne National Laboratory.
 *
 * Copyright 2011 Sandia Corporation.
 * Under the terms of Contract * DE-AC04-94AL85000 with Sandia Corporation,
 * the U.S.  Government retains certain rights in this software.
 *
 * This file was originally part of the Portals SHMEM software package.
 * For license * information, see the LICENSE file in the top level
 * directory of the distribution. */

#ifndef SHMEM_WAIT_H
#define SHMEM_WAIT_H

#include "shmem-internals.h"

#define COMP(type, a, b, ret)                                \
    do {                                                     \
        ret = 0;                                             \
        switch (type) {                                      \
            case SHMEM_CMP_EQ:                               \
                if (a == b) ret = 1;                         \
                break;                                       \
            case SHMEM_CMP_NE:                               \
                if (a != b) ret = 1;                         \
                break;                                       \
            case SHMEM_CMP_GT:                               \
                if (a > b) ret = 1;                          \
                break;                                       \
            case SHMEM_CMP_GE:                               \
                if (a >= b) ret = 1;                         \
                break;                                       \
            case SHMEM_CMP_LT:                               \
                if (a < b) ret = 1;                          \
                break;                                       \
            case SHMEM_CMP_LE:                               \
                if (a <= b) ret = 1;                         \
                break;                                       \
            default:                                         \
                __shmem_abort(type, "invalid comparison");   \
            }                                                \
    } while(0)

#define SHMEM_WAIT(address, value, temp, mpi_type)                          \
    do {                                                                    \
        enum shmem_window_id_e id;                                          \
        shmem_offset_t offset;                                              \
        __shmem_window_offset(address, shmem_world_rank, &id, &offset);     \
                                                                            \
        MPI_Win win;                                                        \
        if (id==SHMEM_SHEAP_WINDOW)                                         \
            win = shmem_sheap_win;                                          \
        else if (id==SHMEM_ETEXT_WINDOW)                                    \
            win = shmem_etext_win;                                          \
                                                                            \
        temp = value;                                                       \
        while (temp == value) {                                             \
            MPI_Fetch_and_op(NULL, &temp, mpi_type, shmem_world_rank,       \
                             offset, MPI_NO_OP, win);                       \
            MPI_Win_flush_local(shmem_world_rank, win);                     \
        }                                                                   \
    } while(0)


#define SHMEM_WAIT_UNTIL(address, cond, value, temp, mpi_type)              \
    do {                                                                    \
        enum shmem_window_id_e id;                                          \
        shmem_offset_t offset; /* not used */                               \
        __shmem_window_offset(address, shmem_world_rank, &id, &offset);     \
                                                                            \
        MPI_Win win;                                                        \
        if (id==SHMEM_SHEAP_WINDOW)                                         \
            win = shmem_sheap_win;                                          \
        else if (id==SHMEM_ETEXT_WINDOW)                                    \
            win = shmem_etext_win;                                          \
                                                                            \
        int cmpret=0;                                                       \
        /* FIXME (?) the next line short-circuits the loop in the case  */  \
        /*           of SHMEM_CMP_EQ, does it not?                      */  \
        temp = value;                                                       \
        COMP(cond, temp, value, cmpret);                                    \
        while (!cmpret) {                                                   \
            MPI_Fetch_and_op(NULL, &temp, mpi_type, shmem_world_rank,       \
                             offset, MPI_NO_OP, win);                       \
            MPI_Win_flush_local(shmem_world_rank, win);                     \
            COMP(cond, temp, value, cmpret);                                \
        }                                                                   \
    } while(0)


#endif // SHMEM_WAIT_H

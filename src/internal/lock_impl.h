/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#ifndef INTERNAL_LOCK_IMPL_H
#define INTERNAL_LOCK_IMPL_H

#include <shmem.h>
#include "oshmpi_impl.h"

#define OSHMPI_LOCK_ROOT_WRANK 0        /* may change after using distributed lock */
#define OSHMPI_LOCK_MSG_TAG 999

typedef struct OSHMPI_lock_s {
    int last;                   /* who is the last on waiting this lock, only meaningful on root */
    int next;                   /* who is waiting on me to release this lock */
} OSHMPI_lock_t;

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_set_lock(long *lockp)
{
    int myid = OSHMPI_global.world_rank + 1;
    int zero = 0;
    int curid = 0;
    OSHMPI_lock_t *lock = (OSHMPI_lock_t *) lockp;
    MPI_Aint lock_last_disp = -1;
    MPI_Aint lock_next_disp = -1;
    MPI_Win win = MPI_WIN_NULL;

    /* TODO: should have a more portable design that does not assume 8-byte long variable */
    OSHMPI_ASSERT(sizeof(long) >= sizeof(OSHMPI_lock_t));

    OSHMPI_translate_win_and_disp((const void *) &lock->last, &win, &lock_last_disp);
    OSHMPI_ASSERT(lock_last_disp >= 0 && win != MPI_WIN_NULL);
    lock_next_disp = lock_last_disp + sizeof(int);

    /* Claim the lock in root process */
    OSHMPI_CALLMPI(MPI_Fetch_and_op
                   (&myid, &curid, MPI_INT, OSHMPI_LOCK_ROOT_WRANK, lock_last_disp, MPI_REPLACE,
                    win));
    OSHMPI_CALLMPI(MPI_Win_flush(OSHMPI_LOCK_ROOT_WRANK, win));

    /* If I am not the last, notify the previous last about me, and wait for release */
    if (curid != zero) {
        OSHMPI_CALLMPI(MPI_Accumulate
                       (&myid, 1, MPI_INT, curid - 1, lock_next_disp, 1, MPI_INT, MPI_REPLACE,
                        win));
        OSHMPI_CALLMPI(MPI_Win_flush(curid - 1, win));
        OSHMPI_CALLMPI(MPI_Recv
                       (&zero, 1, MPI_INT, curid - 1, OSHMPI_LOCK_MSG_TAG, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE));
    }
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_clear_lock(long *lockp)
{
    int myid = OSHMPI_global.world_rank + 1;
    int zero = 0;
    int curid = 0;
    int nextid = 0;
    OSHMPI_lock_t *lock = (OSHMPI_lock_t *) lockp;
    MPI_Aint lock_last_disp = -1;
    MPI_Aint lock_next_disp = -1;
    MPI_Win win = MPI_WIN_NULL;

    OSHMPI_ASSERT(sizeof(long) >= sizeof(OSHMPI_lock_t));

    OSHMPI_translate_win_and_disp((const void *) &lock->last, &win, &lock_last_disp);
    OSHMPI_ASSERT(lock_last_disp >= 0 && win != MPI_WIN_NULL);
    lock_next_disp = lock_last_disp + sizeof(int);

    /* Release the lock in root process if I am the last one holding the lock */
    OSHMPI_CALLMPI(MPI_Compare_and_swap
                   (&zero, &myid, &curid, MPI_INT, OSHMPI_LOCK_ROOT_WRANK, lock_last_disp, win));
    OSHMPI_CALLMPI(MPI_Win_flush(OSHMPI_LOCK_ROOT_WRANK, win));

    /* If I am not the last one, then notify the next that I released */
    if (curid != myid) {
        do {
            OSHMPI_CALLMPI(MPI_Fetch_and_op
                           (&zero, &nextid, MPI_INT, myid - 1, lock_next_disp, MPI_REPLACE, win));
        } while (nextid == 0);
        OSHMPI_CALLMPI(MPI_Send
                       (&myid, 1, MPI_INT, nextid - 1, OSHMPI_LOCK_MSG_TAG, MPI_COMM_WORLD));
    }
}

OSHMPI_STATIC_INLINE_PREFIX int OSHMPI_test_lock(long *lockp)
{
    int myid = OSHMPI_global.world_rank + 1;
    int zero = 0;
    int curid = 0;
    OSHMPI_lock_t *lock = (OSHMPI_lock_t *) lockp;
    MPI_Aint lock_last_disp = -1;
    MPI_Win win = MPI_WIN_NULL;

    OSHMPI_ASSERT(sizeof(long) >= sizeof(OSHMPI_lock_t));

    OSHMPI_translate_win_and_disp((const void *) &lock->last, &win, &lock_last_disp);
    OSHMPI_ASSERT(lock_last_disp >= 0 && win != MPI_WIN_NULL);

    /* Claim the lock in root process, if it is available */
    OSHMPI_CALLMPI(MPI_Compare_and_swap
                   (&myid, &zero, &curid, MPI_INT, OSHMPI_LOCK_ROOT_WRANK, lock_last_disp, win));
    OSHMPI_CALLMPI(MPI_Win_flush(OSHMPI_LOCK_ROOT_WRANK, win));

    if (curid == zero)
        return 0;

    return 1;
}

#endif /* INTERNAL_LOCK_IMPL_H */

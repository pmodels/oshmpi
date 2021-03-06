/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#ifndef INTERNAL_LOCK_IMPL_H
#define INTERNAL_LOCK_IMPL_H

#include <shmem.h>
#include <limits.h>
#include "oshmpi_impl.h"

#define OSHMPI_LOCK_ROOT_WRANK 0        /* may change after using distributed lock */

#define NEXT_MASK (INT_MAX)
#define SIGNAL_MASK ((unsigned int)INT_MAX+1)
#define NEXT(a) (int)(a & NEXT_MASK)
#define SIGNAL(a) (int)(a & SIGNAL_MASK)

typedef struct OSHMPI_lock_s {
    int last;                   /* who is the last on waiting this lock, only meaningful on root */
    unsigned int next;          /* sign bit: signal of lock release;
                                 * other bits: next pe who is waiting on me to release this lock */
} OSHMPI_lock_t;

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_set_lock(long *lockp)
{
    int myid = OSHMPI_global.team_world_my_pe + 1;
    int zero = 0;
    int curid = 0;
    OSHMPI_lock_t *lock = (OSHMPI_lock_t *) lockp;
    MPI_Aint lock_last_disp = -1;
    MPI_Aint lock_next_disp = -1;
    OSHMPI_ictx_t *ictx = NULL;
    OSHMPI_sobj_attr_t *sobj_attr = NULL;
    unsigned int signal = 0, next = (unsigned int) myid;

    /* TODO: should have a more portable design that does not assume 8-byte long variable */
    OSHMPI_ASSERT(sizeof(long) >= sizeof(OSHMPI_lock_t));

    OSHMPI_sobj_query_attr_ictx(SHMEM_CTX_DEFAULT, (const void *) &lock->last,
                                OSHMPI_LOCK_ROOT_WRANK, &sobj_attr, &ictx);
    OSHMPI_ASSERT(sobj_attr && ictx);
    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, (const void *) &lock->last, OSHMPI_LOCK_ROOT_WRANK,
                                    OSHMPI_ICTX_DISP_MODE(ictx), &lock_last_disp);
    OSHMPI_ASSERT(lock_last_disp >= 0);

    lock_next_disp = lock_last_disp + sizeof(int);

    /* Reset my local bits. No one accesses to my next bits now. */
    lock->next = 0;
    OSHMPI_CALLMPI(MPI_Win_sync(ictx->win));

    /* Claim the lock in root process */
    OSHMPI_CALLMPI(MPI_Fetch_and_op
                   (&myid, &curid, MPI_INT, OSHMPI_LOCK_ROOT_WRANK, lock_last_disp, MPI_REPLACE,
                    ictx->win));
    OSHMPI_CALLMPI(MPI_Win_flush(OSHMPI_LOCK_ROOT_WRANK, ictx->win));
    OSHMPI_DBGMSG("%s %p, curid=%d\n", (curid == zero) ? "locked" : "queued", lockp, curid - 1);

    /* If I am not the last, notify the previous last about me, and wait for release */
    if (curid != zero) {
        OSHMPI_CALLMPI(MPI_Accumulate(&next, 1, MPI_UNSIGNED, curid - 1, lock_next_disp,
                                      1, MPI_UNSIGNED, MPI_BOR, ictx->win));
        OSHMPI_CALLMPI(MPI_Win_flush(curid - 1, ictx->win));

        /* Wait till received release signal of this lock.
         * Do not reset, we will reset at next set_lock call. */
        while (1) {
            OSHMPI_CALLMPI(MPI_Fetch_and_op
                           (NULL, &signal, MPI_UNSIGNED, OSHMPI_global.team_world_my_pe,
                            lock_next_disp, MPI_NO_OP, ictx->win));
            OSHMPI_CALLMPI(MPI_Win_flush(OSHMPI_global.team_world_my_pe, ictx->win));
            if (SIGNAL(signal))
                break;
            OSHMPI_am_cb_progress();
        }
        OSHMPI_DBGMSG("released by others, locked %p\n", lockp);
    }
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_clear_lock(long *lockp)
{
    int myid = OSHMPI_global.team_world_my_pe + 1;
    int zero = 0;
    int curid = 0, nextid = 0;
    OSHMPI_lock_t *lock = (OSHMPI_lock_t *) lockp;
    MPI_Aint lock_last_disp = -1;
    MPI_Aint lock_next_disp = -1;
    OSHMPI_ictx_t *ictx = NULL;
    OSHMPI_sobj_attr_t *sobj_attr = NULL;
    unsigned int next = 0, signal = SIGNAL_MASK;

    OSHMPI_ASSERT(sizeof(long) >= sizeof(OSHMPI_lock_t));

    OSHMPI_sobj_query_attr_ictx(SHMEM_CTX_DEFAULT, (const void *) &lock->last,
                                OSHMPI_LOCK_ROOT_WRANK, &sobj_attr, &ictx);
    OSHMPI_ASSERT(sobj_attr && ictx);
    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, (const void *) &lock->last, OSHMPI_LOCK_ROOT_WRANK,
                                    OSHMPI_ICTX_DISP_MODE(ictx), &lock_last_disp);
    OSHMPI_ASSERT(lock_last_disp >= 0);

    lock_next_disp = lock_last_disp + sizeof(int);

    /* Release the lock in root process if I am the last one holding the lock */
    OSHMPI_CALLMPI(MPI_Compare_and_swap
                   (&zero, &myid, &curid, MPI_INT, OSHMPI_LOCK_ROOT_WRANK, lock_last_disp,
                    ictx->win));
    OSHMPI_CALLMPI(MPI_Win_flush(OSHMPI_LOCK_ROOT_WRANK, ictx->win));
    OSHMPI_DBGMSG("released lock %p, curid=%d\n", lockp, curid - 1);

    /* If I am not the last one, then notify the next that I released */
    if (curid != myid) {
        while (1) {
            OSHMPI_CALLMPI(MPI_Fetch_and_op
                           (NULL, &next, MPI_UNSIGNED, OSHMPI_global.team_world_my_pe,
                            lock_next_disp, MPI_NO_OP, ictx->win));
            OSHMPI_CALLMPI(MPI_Win_flush(OSHMPI_global.team_world_my_pe, ictx->win));
            nextid = NEXT(next);
            if (nextid != 0)
                break;
            OSHMPI_am_cb_progress();
        }

        /* Reset my local bits. No one accesses to my next bits now. */
        lock->next = 0;
        OSHMPI_CALLMPI(MPI_Win_sync(ictx->win));

        /* Set next PE's signal bit */
        OSHMPI_CALLMPI(MPI_Accumulate(&signal, 1, MPI_UNSIGNED, nextid - 1, lock_next_disp, 1,
                                      MPI_UNSIGNED, MPI_BOR, ictx->win));
        OSHMPI_CALLMPI(MPI_Win_flush(nextid - 1, ictx->win));
        OSHMPI_DBGMSG("pass lock %p to %d\n", lockp, nextid - 1);
    }
}

OSHMPI_STATIC_INLINE_PREFIX int OSHMPI_test_lock(long *lockp)
{
    int myid = OSHMPI_global.team_world_my_pe + 1;
    int zero = 0;
    int curid = 0;
    OSHMPI_lock_t *lock = (OSHMPI_lock_t *) lockp;
    MPI_Aint lock_last_disp = -1;
    OSHMPI_ictx_t *ictx = NULL;
    OSHMPI_sobj_attr_t *sobj_attr = NULL;

    OSHMPI_ASSERT(sizeof(long) >= sizeof(OSHMPI_lock_t));

    OSHMPI_sobj_query_attr_ictx(SHMEM_CTX_DEFAULT, (const void *) &lock->last,
                                OSHMPI_LOCK_ROOT_WRANK, &sobj_attr, &ictx);
    OSHMPI_ASSERT(sobj_attr && ictx);
    OSHMPI_sobj_trans_vaddr_to_disp(sobj_attr, (const void *) &lock->last, OSHMPI_LOCK_ROOT_WRANK,
                                    OSHMPI_ICTX_DISP_MODE(ictx), &lock_last_disp);
    OSHMPI_ASSERT(lock_last_disp >= 0);

    /* Claim the lock in root process, if it is available */
    OSHMPI_CALLMPI(MPI_Compare_and_swap
                   (&myid, &zero, &curid, MPI_INT, OSHMPI_LOCK_ROOT_WRANK, lock_last_disp,
                    ictx->win));
    OSHMPI_CALLMPI(MPI_Win_flush(OSHMPI_LOCK_ROOT_WRANK, ictx->win));

    if (curid == zero) {
        OSHMPI_DBGMSG("locked %p, curid=%d\n", lockp, curid - 1);
        return 0;
    }
    return 1;
}

#endif /* INTERNAL_LOCK_IMPL_H */

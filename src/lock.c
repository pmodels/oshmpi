#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>

#include "shmem-internals.h"
#include "lock.h"

int *lock_base = NULL;

void
_allock (MPI_Comm comm)
{
  MPI_Win_allocate (4 * sizeof (int), sizeof (int), MPI_INFO_NULL,
		    comm, &lock_base, &lock_win);
  lock_base[NEXT_DISP] = -1;
  lock_base[PREV_DISP] = -1;
  lock_base[TAIL_DISP] = -1;
  lock_base[LOCK_DISP] = -1;
  MPI_Win_lock_all (TAIL, lock_win);
  return;
}

void
_deallock (void)
{
  MPI_Win_unlock_all (lock_win);
  MPI_Win_free (&lock_win);
  return;
}

void
_lock (long *lockp)
{
  MPI_Status status;
  lock_t *lock = (lock_t *) lockp;
  /* Replace myself with the last tail */
  MPI_Fetch_and_op (&shmem_world_rank, &(lock->prev), MPI_INT, TAIL,
		    TAIL_DISP, MPI_REPLACE, lock_win);
  MPI_Win_flush (TAIL, lock_win);

  /* Previous proc holding lock will eventually notify */
  if (lock->prev != -1)
    {
      /* Send my shmem_world_rank to previous proc's next */
      MPI_Accumulate (&shmem_world_rank, 1, MPI_INT, lock->prev, NEXT_DISP,
		      1, MPI_INT, MPI_REPLACE, lock_win);
      MPI_Win_flush (lock->prev, lock_win);
      MPI_Probe (lock->prev, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
  /* Hold lock */
  lock_base[LOCK_DISP] = 1;
  MPI_Win_sync (lock_win);

  return;
}

void
_unlock (long *lockp)
{
  lock_t *lock = (lock_t *) lockp;
  /* Determine my next process */
  MPI_Fetch_and_op (NULL, &(lock->next), MPI_INT, shmem_world_rank,
		    NEXT_DISP, MPI_NO_OP, lock_win);
  MPI_Win_flush (shmem_world_rank, lock_win);

  if (lock->next != -1)
    {
      MPI_Send (&shmem_world_rank, 1, MPI_INT, lock->next, 999, MPI_COMM_WORLD);
    }
  /* Release lock */
  lock_base[LOCK_DISP] = -1;
  MPI_Win_sync (lock_win);

  return;
}

int
_trylock (long *lockp)
{
  int is_locked = -1, nil = -1;
  lock_t *lock = (lock_t *) lockp;
  lock->prev = -1;
  /* Get the last tail, if -1 replace with me */
  MPI_Compare_and_swap (&shmem_world_rank, &nil, &(lock->prev), MPI_INT,
			TAIL, TAIL_DISP, lock_win);
  MPI_Win_flush (TAIL, lock_win);
  /* Find if the last proc is holding lock */
  if (lock->prev != -1)
    {
      MPI_Fetch_and_op (NULL, &is_locked, MPI_INT, lock->prev,
			LOCK_DISP, MPI_NO_OP, lock_win);
      MPI_Win_flush (lock->prev, lock_win);

      if (is_locked)
	return 0;
    }
  /* Add myself in tail */
  MPI_Fetch_and_op (&shmem_world_rank, &(lock->prev), MPI_INT, TAIL,
		    TAIL_DISP, MPI_REPLACE, lock_win);
  MPI_Win_flush (TAIL, lock_win);
  /* Hold lock */
  lock_base[LOCK_DISP] = 1;
  MPI_Win_sync (lock_win);

  return 1;
}

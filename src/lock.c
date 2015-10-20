#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>

#include "shmem-internals.h"
#include "oshmpi-mcs-lock.h"

#define NEXT_DISP 1
#define PREV_DISP 0
#define TAIL_DISP 2 //Only has meaning in PE 0
#define LOCK_DISP 3
#define TAIL 	  0

MPI_Win oshmpi_lock_win = MPI_WIN_NULL;
int * oshmpi_lock_base = NULL;

void oshmpi_allock(MPI_Comm comm)
{
  MPI_Info lock_info=MPI_INFO_NULL;
  MPI_Info_create(&lock_info);

  /* We define the sheap size to be symmetric and assume it for the global static data. */
  MPI_Info_set(lock_info, "same_size", "true");

  MPI_Win_allocate (4 * sizeof (int), sizeof (int), lock_info,
		    comm, &oshmpi_lock_base, &oshmpi_lock_win);
  oshmpi_lock_base[NEXT_DISP] = -1;
  oshmpi_lock_base[PREV_DISP] = -1;
  oshmpi_lock_base[TAIL_DISP] = -1;
  oshmpi_lock_base[LOCK_DISP] = -1;
  MPI_Win_lock_all (TAIL, oshmpi_lock_win);

  MPI_Info_free(&lock_info);

  return;
}

void oshmpi_deallock(void)
{
  MPI_Win_unlock_all (oshmpi_lock_win);
  MPI_Win_free (&oshmpi_lock_win);
  return;
}

void oshmpi_lock(long * lockp)
{
  MPI_Status status;
  oshmpi_lock_t *lock = (oshmpi_lock_t *) lockp;
  /* Replace myself with the last tail */
  MPI_Fetch_and_op (&shmem_world_rank, &(lock->prev), MPI_INT, TAIL,
		    TAIL_DISP, MPI_REPLACE, oshmpi_lock_win);
  MPI_Win_flush (TAIL, oshmpi_lock_win);

  /* Previous proc holding lock will eventually notify */
  if (lock->prev != -1)
    {
      /* Send my shmem_world_rank to previous proc's next */
      MPI_Accumulate (&shmem_world_rank, 1, MPI_INT, lock->prev, NEXT_DISP,
		      1, MPI_INT, MPI_REPLACE, oshmpi_lock_win);
      MPI_Win_flush (lock->prev, oshmpi_lock_win);
      MPI_Probe (lock->prev, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
  /* Hold lock */
  oshmpi_lock_base[LOCK_DISP] = 1;
  MPI_Win_sync (oshmpi_lock_win);

  return;
}

void oshmpi_unlock(long * lockp)
{
  oshmpi_lock_t *lock = (oshmpi_lock_t *) lockp;
  /* Determine my next process */
  MPI_Fetch_and_op (NULL, &(lock->next), MPI_INT, shmem_world_rank,
		    NEXT_DISP, MPI_NO_OP, oshmpi_lock_win);
  MPI_Win_flush (shmem_world_rank, oshmpi_lock_win);

  if (lock->next != -1)
    {
      MPI_Send (&shmem_world_rank, 1, MPI_INT, lock->next, 999, MPI_COMM_WORLD);
    }
  /* Release lock */
  oshmpi_lock_base[LOCK_DISP] = -1;
  MPI_Win_sync (oshmpi_lock_win);

  return;
}

int oshmpi_trylock(long * lockp)
{
  int is_locked = -1, nil = -1;
  oshmpi_lock_t *lock = (oshmpi_lock_t *) lockp;
  lock->prev = -1;
  /* Get the last tail, if -1 replace with me */
  MPI_Compare_and_swap (&shmem_world_rank, &nil, &(lock->prev), MPI_INT,
			TAIL, TAIL_DISP, oshmpi_lock_win);
  MPI_Win_flush (TAIL, oshmpi_lock_win);
  /* Find if the last proc is holding lock */
  if (lock->prev != -1)
    {
      MPI_Fetch_and_op (NULL, &is_locked, MPI_INT, lock->prev,
			LOCK_DISP, MPI_NO_OP, oshmpi_lock_win);
      MPI_Win_flush (lock->prev, oshmpi_lock_win);

      if (is_locked)
	return 0;
    }
  /* Add myself in tail */
  MPI_Fetch_and_op (&shmem_world_rank, &(lock->prev), MPI_INT, TAIL,
		    TAIL_DISP, MPI_REPLACE, oshmpi_lock_win);
  MPI_Win_flush (TAIL, oshmpi_lock_win);
  /* Hold lock */
  oshmpi_lock_base[LOCK_DISP] = 1;
  MPI_Win_sync (oshmpi_lock_win);

  return 1;
}

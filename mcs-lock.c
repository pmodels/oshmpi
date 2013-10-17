#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <strings.h>

#include "mcs-lock.h"
#include "shmem-internals.h"

/* Allocate a qnode */
void alloc_qnode(void) 
{
	/* Allocate window */
	MPI_Win_allocate(2*sizeof(long), sizeof(long), MPI_INFO_NULL, MPI_COMM_WORLD, &qnode, &qnode_win);
	qnode[0] = -1; /* Tail rank */
	qnode[1] = -1; /* Pointer to successor in queue (Successor's rank) */
	/* Lock the window for shared access to all targets */
	MPI_Win_lock_all(0, qnode_win);
	return;
}

void dealloc_qnode(void)
{
	MPI_Win_unlock_all (qnode_win);
	MPI_Win_free (&qnode_win);
	return;
}

int acquire_mcslock(long * lock_addr) 
{
	alloc_qnode();
	tail_ptr.disp = (MPI_Aint)lock_addr;
	tail_ptr.procid = 0; /* procid holds the tail ptr */
	qnode[MCS_MTX_ELEM_DISP] = -1;
	MPI_Win_sync(qnode_win);
	/* Set predecessor */	
	qnode_ptr_t pred = nil;
	/* Update tail_procid and get previous tail */
	/* TODO disp must not be hardcoded */
	MPI_Fetch_and_op (&shmem_world_rank, &pred.procid, MPI_INT, tail_ptr.procid, MCS_MTX_TAIL_DISP, MPI_REPLACE, qnode_win);
	MPI_Win_flush(tail_ptr.procid, qnode_win);
	/* If there was a previous tail, update it's next ptr */
	if (pred.procid != -1) {
		MPI_Accumulate(&shmem_world_rank, 1, MPI_INT, pred.procid, MCS_MTX_ELEM_DISP, 1, MPI_INT, MPI_REPLACE, qnode_win);
		MPI_Win_flush(pred.procid, qnode_win);
		/* Spin until */
		MPI_Recv(NULL, 0, MPI_BYTE, pred.procid, MCS_MUTEX_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	return MPI_SUCCESS;
}

int release_mcslock(long * lock_addr) 
{
	int next, nnil=-1, flag, tail;
	tail_ptr.disp = (MPI_Aint)lock_addr;

	/* Read my next pointer.  FOP is used since another process may write to
	 * this location concurrent with this read. */
	MPI_Fetch_and_op(&nnil, &next, MPI_INT, shmem_world_rank, MCS_MTX_ELEM_DISP, MPI_NO_OP, qnode_win);
	MPI_Win_flush(shmem_world_rank, qnode_win);

	if ( next == -1) {
		/* Check if we are the at the tail of the lock queue.  If so, we're
		 * done.  If not, we need to send notification. */
		MPI_Compare_and_swap(&nnil, &shmem_world_rank, &tail, MPI_INT, tail_ptr.procid, MCS_MTX_TAIL_DISP, qnode_win);
		MPI_Win_flush(tail_ptr.procid, qnode_win);

		if (tail != shmem_world_rank) {
			assert(tail >= 0 && tail < shmem_world_size);
			/* Traverse till the end */
			for (;;) {
				MPI_Fetch_and_op(&nnil, &next, MPI_INT, shmem_world_rank, MCS_MTX_ELEM_DISP, MPI_NO_OP, qnode_win);
				MPI_Win_flush(shmem_world_rank, qnode_win);
				if (next != -1) break;
				MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
			}
		}
	}

	/* Notify the next waiting process */
	if (next != -1) {
		MPI_Send(NULL, 0, MPI_BYTE, next, MCS_MUTEX_TAG, MPI_COMM_WORLD);
	}

	return MPI_SUCCESS;
}

int test_mcslock(long * lock_addr, int * success)
{
	int nnil, tail = -1;
	qnode[MCS_MTX_ELEM_DISP] = -1;
	MPI_Win_sync(qnode_win);
	/* Set predecessor */	
	tail_ptr.disp = (MPI_Aint)lock_addr;
	
	/* Check if the lock is available and claim it if it is. */
	MPI_Compare_and_swap(&shmem_world_rank, &nnil, &tail, MPI_INT, tail_ptr.procid, MCS_MTX_TAIL_DISP, qnode_win);
	MPI_Win_flush(tail_ptr.procid, qnode_win);

	/* If the old tail was -1, we have claimed the mutex */
	*success = ((tail == nnil) ? 0 : 1);
	return MPI_SUCCESS;
}

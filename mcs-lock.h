/*
  slightly modified from mcs-mutex:
http://trac.mpich.org/projects/mpich/browser/test/mpi/rma/mcs-mutex.c?rev=89407ecc99f5b08a6fd44132ae44faa99dc9a71e 
 */
#ifndef MCS_LOCK_H
#define MCS_LOCK_H
#include <mpi.h>

#define MCS_MUTEX_TAG 100
#define MCS_MTX_ELEM_DISP 0
#define MCS_MTX_TAIL_DISP 1

MPI_Win qnode_win;
long * qnode;

/* Pointer to list element */
typedef struct qnode_ptr_s {
	int      procid;
	MPI_Aint disp;
} qnode_ptr_t;

static const qnode_ptr_t nil = { -1, (MPI_Aint)MPI_BOTTOM };
qnode_ptr_t tail_ptr;

/* Allocate a qnode */
void alloc_qnode(void); 
void dealloc_qnode(void);
int acquire_mcslock(long * lock_addr); 
int release_mcslock(long * lock_addr); 
int test_mcslock(long * lock_addr, int * success);
#endif /* MCS_LOCK_H_INCLUDED */

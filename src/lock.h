#ifndef LOCK_H
#define LOCK_H

#include "shmem.h"

/* MPI Lock */
MPI_Win lock_win;
extern int *lock_base;

typedef struct lock_s
{
  int prev;
  int next;
} lock_t;

#define NEXT_DISP 1
#define PREV_DISP 0
#define TAIL_DISP 2 //Only has meaning in PE 0
#define LOCK_DISP 3
#define TAIL 	  0

void _allock (MPI_Comm comm);
void _deallock (void);
void _lock (long *lockp);
void _unlock (long *lockp);
int _trylock (long *lockp);

#endif /* LOCK_H */

#ifndef LOCK_H
#define LOCK_H

#include "shmem-internals.h"

/* MPI Lock */
extern MPI_Win oshmpi_lock_win;
extern int * oshmpi_lock_base;

typedef struct oshmpi_lock_s
{
  int prev;
  int next;
} oshmpi_lock_t;

void oshmpi_allock(MPI_Comm comm);
void oshmpi_deallock(void);
void oshmpi_lock(long * lockp);
void oshmpi_unlock(long * lockp);
int  oshmpi_trylock(long * lockp);

#endif /* LOCK_H */

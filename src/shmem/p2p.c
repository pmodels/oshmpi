/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include "oshmpi_impl.h"

/* Deprecated APIs start */
void shmem_wait_until(long *ivar, int cmp, long cmp_value)
{
    /* Does not return until ivar satisfies the condition implied by cmp and cmp_value. */
    OSHMPI_WAIT_UNTIL(ivar, cmp, cmp_value, long, MPI_LONG);
}

void shmem_wait(long *ivar, long cmp_value)
{
    /* Return when ivar is no longer equal to cmp_value. */
    OSHMPI_WAIT_UNTIL(ivar, SHMEM_CMP_NE, cmp_value, long, MPI_LONG);
}

void shmem_short_wait(short *ivar, short cmp_value)
{
    /* Return when ivar is no longer equal to cmp_value. */
    OSHMPI_WAIT_UNTIL(ivar, SHMEM_CMP_NE, cmp_value, short, MPI_SHORT);
}

void shmem_int_wait(int *ivar, int cmp_value)
{
    /* Return when ivar is no longer equal to cmp_value. */
    OSHMPI_WAIT_UNTIL(ivar, SHMEM_CMP_NE, cmp_value, int, MPI_INT);
}

void shmem_long_wait(long *ivar, long cmp_value)
{
    /* Return when ivar is no longer equal to cmp_value. */
    OSHMPI_WAIT_UNTIL(ivar, SHMEM_CMP_NE, cmp_value, long, MPI_LONG);
}

void shmem_longlong_wait(long long *ivar, long long cmp_value)
{
    /* Return when ivar is no longer equal to cmp_value. */
    OSHMPI_WAIT_UNTIL(ivar, SHMEM_CMP_NE, cmp_value, long long, MPI_LONG_LONG);
}

/* Deprecated APIs end */

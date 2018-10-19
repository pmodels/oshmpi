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
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_wait(long *ivar, long cmp_value)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_short_wait(short *ivar, short cmp_value)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_int_wait(int *ivar, int cmp_value)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_long_wait(long *ivar, long cmp_value)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_longlong_wait(long long *ivar, long long cmp_value)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

/* Deprecated APIs end */

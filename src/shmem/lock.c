/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include "oshmpi_impl.h"

void shmem_clear_lock(long *lock)
{
    OSHMPI_clear_lock(lock);
}

void shmem_set_lock(long *lock)
{
    OSHMPI_set_lock(lock);
}

int shmem_test_lock(long *lock)
{
    return OSHMPI_test_lock(lock);
}

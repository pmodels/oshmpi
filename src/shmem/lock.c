/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include "oshmpi_impl.h"

void shmem_clear_lock(long *lock)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

void shmem_set_lock(long *lock)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
}

int shmem_test_lock(long *lock)
{
    OSHMPI_ERR_ABORT("Unsupported function: %s\n", __FUNCTION__);
    return 0;
}

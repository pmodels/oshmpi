/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>

/* Spec v1.4 section 9.12: They are not required to be supported by
 * implementing them as no-ops and where used, they may have no effect on
 * cache line states. */

void shmem_clear_cache_inv(void)
{
}

void shmem_set_cache_inv(void)
{
}

void shmem_clear_cache_line_inv(void *dest)
{
}

void shmem_set_cache_line_inv(void *dest)
{
}

void shmem_udcflush(void)
{
}

void shmem_udcflush_line(void *dest)
{
}

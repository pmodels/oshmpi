/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmemx.h>
#include "oshmpi_impl.h"

int shmemx_query_interoperability(int property)
{
    int result = 0;

    switch (property) {
        case SHMEMX_PROGRESS_MPI:
            result = 1;
            break;
        default:
            break;
    }
    return result;
}

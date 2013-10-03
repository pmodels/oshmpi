/* -*- C -*-
 *
 * Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
 * 
 * This file is part of the Portals SHMEM software package. For license
 * information, see the LICENSE file in the top level directory of the
 * distribution.
 *
 */

#include <mpi.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/param.h>
#include <stdio.h>
#include <limits.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

#include "shmem.h"
#include "shmem_internal.h"
#include "shmem_comm.h"

int
shmem_transport_mpi3_init(long eager_size)
{
    ptl_process_t my_id;
    int ret;
    ptl_ni_limits_t ni_req_limits;

    /* Initialize Portals */
    ret = PtlInit();
    if (PTL_OK != ret) {
        fprintf(stderr, "ERROR: PtlInit failed: %d\n", ret);
        return 1;
    }

    return 0;
}



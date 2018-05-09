/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef OSHMPI_IMPL_H
#define OSHMPI_IMPL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <oshmpiconf.h>

#include "oshmpi_util.h"

typedef struct {
    int is_initialized;
    int world_rank;
    int world_size;
} OSHMPI_global_t;

extern OSHMPI_global_t OSHMPI_global;

int OSHMPI_initialize_thread(int required, int *provided);
int OSHMPI_finalize(void);

#endif /* OSHMPI_IMPL_H */

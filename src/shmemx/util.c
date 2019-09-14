/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include <shmemx.h>
#include "oshmpi_impl.h"

OSHMPI_TIMER_EXTERN_DECL(iput_dtype_create);
OSHMPI_TIMER_EXTERN_DECL(iput_comm);
OSHMPI_TIMER_EXTERN_DECL(iput_dtype_free);
OSHMPI_TIMER_EXTERN_DECL(quiet);

void shmemx_print_timer(void)
{
#ifdef OSHMPI_ENABLE_TIMER
    OSHMPI_PRINT_TIMER(iput_dtype_create);
    OSHMPI_PRINT_TIMER(iput_comm);
    OSHMPI_PRINT_TIMER(iput_dtype_free);
    OSHMPI_PRINT_TIMER(quiet);
#else
    OSHMPI_PRINTF("OSHMPI timer is disabled, recompile with --enable-timer to enable.\n");
#endif
}

void shmemx_timer_reset(void)
{
    OSHMPI_TIMER_RESET(iput_dtype_create);
    OSHMPI_TIMER_RESET(iput_comm);
    OSHMPI_TIMER_RESET(iput_dtype_free);
    OSHMPI_PRINT_TIMER(quiet);
}

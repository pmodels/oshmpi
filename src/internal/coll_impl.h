/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_COLL_IMPL_H
#define INTERNAL_COLL_IMPL_H

/* Block until all PEs arrive at the barrier and all local updates
 * and remote memory updates on the default context are completed. */
static inline void OSHMPI_barrier_all(void)
{
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_heap_win));
    /* TODO: flush etext */
    OSHMPI_CALLMPI(MPI_Barrier(OSHMPI_global.comm_world));
}

#endif /* INTERNAL_COLL_IMPL_H */

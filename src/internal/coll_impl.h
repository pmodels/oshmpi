/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_COLL_IMPL_H
#define INTERNAL_COLL_IMPL_H

static inline void OSHMPI_coll_initialize(void)
{
    OSHMPI_global.comm_cache_list.nobjs = 0;
    OSHMPI_global.comm_cache_list.head = NULL;
}

static inline void OSHMPI_coll_finalize(void)
{
    OSHMPI_comm_cache_obj_t *cobj, *tmp;

    /* Release all cached comm */
    LL_FOREACH_SAFE(OSHMPI_global.comm_cache_list.head, cobj, tmp) {
        LL_DELETE(OSHMPI_global.comm_cache_list.head, cobj);
        OSHMPI_CALLMPI(MPI_Comm_free(&cobj->comm));
        OSHMPIU_free(cobj);
        OSHMPI_global.comm_cache_list.nobjs--;
    }
    OSHMPI_ASSERT(OSHMPI_global.comm_cache_list.nobjs == 0);
}

/* Cache a newly created comm.
 * Note that we have to cache all comms to ensure it is cached on all involved pes.
 * However, we expect that the amount of different active sets will be small.*/
static inline void coll_set_comm_cache(int PE_start, int logPE_stride, int PE_size, MPI_Comm comm)
{
    OSHMPI_comm_cache_obj_t *cobj = NULL;

    cobj = OSHMPIU_malloc(sizeof(OSHMPI_comm_cache_obj_t));
    OSHMPI_ASSERT(cobj);

    /* Set new comm */
    cobj->pe_start = PE_start;
    cobj->pe_stride = logPE_stride;
    cobj->pe_size = PE_size;
    cobj->comm = comm;

    /* Insert in head, O(1) */
    LL_PREPEND(OSHMPI_global.comm_cache_list.head, cobj);
    OSHMPI_global.comm_cache_list.nobjs++;
}

/* Find if cached comm already exists. */
static inline int coll_find_comm_cache(int PE_start, int logPE_stride, int PE_size, MPI_Comm * comm)
{
    int found = 0;
    OSHMPI_comm_cache_obj_t *cobj = OSHMPI_global.comm_cache_list.head;

    LL_FOREACH(OSHMPI_global.comm_cache_list.head, cobj) {
        if (cobj->pe_start == PE_start && cobj->pe_stride == logPE_stride
            && cobj->pe_size == PE_size) {
            found = 1;
            *comm = cobj->comm;
            break;
        }
    }
    return found;
}

static inline void coll_acquire_comm(int PE_start, int logPE_stride, int PE_size, MPI_Comm * comm)
{
    /* Fast path: comm_world */
    if (PE_start == 0 && logPE_stride == 0 && PE_size == OSHMPI_global.world_size) {
        *comm = OSHMPI_global.comm_world;
        OSHMPI_DBGMSG("active_set[%d,%d,%d]=>comm_world 0x%lx returned.\n",
                      PE_start, logPE_stride, PE_size, (unsigned long) *comm);
        return;
    }

    /* Fast path: return a cached comm if found */
    if (coll_find_comm_cache(PE_start, logPE_stride, PE_size, comm)) {
        OSHMPI_DBGMSG("active_set[%d,%d,%d]=>cached comm 0x%lx returned.\n",
                      PE_start, logPE_stride, PE_size, (unsigned long) *comm);
        return;
    }

    /* Slow path: create a new communicator and cache it */
    MPI_Group strided_group = MPI_GROUP_NULL;

    /* List of processes in the group that will be created. */
    int *pe_list = NULL;
    pe_list = (int *) OSHMPIU_malloc(PE_size * sizeof(int));
    OSHMPI_ASSERT(pe_list != NULL);

    /* Implement 2^pe_logs with bitshift. */
    const int pe_stride = 1 << logPE_stride;
    for (int i = 0; i < PE_size; i++)
        pe_list[i] = PE_start + i * pe_stride;

    OSHMPI_CALLMPI(MPI_Group_incl
                   (OSHMPI_global.comm_world_group, PE_size, pe_list, &strided_group));
    /* Only collective on the strided_group. */
    OSHMPI_CALLMPI(MPI_Comm_create_group
                   (OSHMPI_global.comm_world, strided_group, PE_start /* tag */ , comm));

    OSHMPI_CALLMPI(MPI_Group_free(&strided_group));
    OSHMPIU_free(pe_list);

    coll_set_comm_cache(PE_start, logPE_stride, PE_size, *comm);
    OSHMPI_DBGMSG("new active_set[%d,%d,%d]=>comm 0x%lx created and cached.\n",
                  PE_start, logPE_stride, PE_size, (unsigned long) *comm);
}

/* Block until all PEs arrive at the barrier and all local updates
 * and remote memory updates on the default context are completed. */
static inline void OSHMPI_barrier_all(void)
{
    /* Ensure ordered delivery of all outstanding Put, AMO, and nonblocking Put */
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_heap_win));

    /* Ensure ordered delivery of memory store */
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_heap_win));

    /* TODO: flush etext */

    OSHMPI_CALLMPI(MPI_Barrier(OSHMPI_global.comm_world));
}

static inline void OSHMPI_barrier(int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    MPI_Comm comm = MPI_COMM_NULL;

    /* Ensure ordered delivery of all outstanding Put, AMO, and nonblocking Put */
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_heap_win));

    /* Ensure ordered delivery of memory store */
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_heap_win));

    coll_acquire_comm(PE_start, logPE_stride, PE_size, &comm);
    OSHMPI_CALLMPI(MPI_Barrier(comm));
}

#endif /* INTERNAL_COLL_IMPL_H */

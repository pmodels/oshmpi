/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_COLL_IMPL_H
#define INTERNAL_COLL_IMPL_H

typedef struct OSHMPI_comm_cache_obj {
    OSHMPI_MEMPOOL_OBJ_HEADER;
    int pe_start;
    int pe_stride;
    int pe_size;
    MPI_Comm comm;
    MPI_Group group;            /* Cached in case we need to translate root rank. */
    struct OSHMPI_comm_cache_obj *next;
} OSHMPI_comm_cache_obj_t;

typedef struct OSHMPI_comm_cache {
    OSHMPI_comm_cache_obj_t *head;      /* List of cached communicator objects */
    int nobjs;
    OSHMPI_mempool_t mempool;
    OSHMPIU_thread_cs_t thread_cs;
} OSHMPI_comm_cache_t;

extern OSHMPI_comm_cache_t OSHMPI_coll_comm_cache;

/* Cache a newly created comm.
 * Note that we have to cache all comms to ensure it is cached on all involved pes.
 * However, we expect that the amount of different active sets will be small.*/
OSHMPI_STATIC_INLINE_PREFIX void coll_set_comm_cache(int PE_start, int logPE_stride, int PE_size,
                                                     MPI_Comm comm, MPI_Group group)
{
    OSHMPI_comm_cache_obj_t *cobj = NULL;

    OSHMPI_THREAD_ENTER_CS(&OSHMPI_coll_comm_cache.thread_cs);
    cobj = OSHMPIU_mempool_alloc_obj(&OSHMPI_coll_comm_cache.mempool);

    /* Set new comm */
    cobj->pe_start = PE_start;
    cobj->pe_stride = logPE_stride;
    cobj->pe_size = PE_size;
    cobj->comm = comm;
    cobj->group = group;

    /* Insert in head, O(1) */
    LL_PREPEND(OSHMPI_coll_comm_cache.head, cobj);
    OSHMPI_coll_comm_cache.nobjs++;
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_coll_comm_cache.thread_cs);
}

/* Find if cached comm already exists. */
OSHMPI_STATIC_INLINE_PREFIX int coll_find_comm_cache(int PE_start, int logPE_stride, int PE_size,
                                                     MPI_Comm * comm, MPI_Group * group)
{
    int found = 0;
    OSHMPI_comm_cache_obj_t *cobj = NULL;

    OSHMPI_THREAD_ENTER_CS(&OSHMPI_coll_comm_cache.thread_cs);
    cobj = OSHMPI_coll_comm_cache.head;
    LL_FOREACH(OSHMPI_coll_comm_cache.head, cobj) {
        if (cobj->pe_start == PE_start && cobj->pe_stride == logPE_stride
            && cobj->pe_size == PE_size) {
            found = 1;
            *comm = cobj->comm;
            *group = cobj->group;
            break;
        }
    }
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_coll_comm_cache.thread_cs);
    return found;
}

OSHMPI_STATIC_INLINE_PREFIX void coll_acquire_comm(int PE_start, int logPE_stride, int PE_size,
                                                   MPI_Comm * comm)
{
    MPI_Group strided_group = MPI_GROUP_NULL;

    /* Fast path: comm_world */
    if (PE_start == 0 && logPE_stride == 0 && PE_size == OSHMPI_global.world_size) {
        *comm = OSHMPI_global.comm_world;
        OSHMPI_DBGMSG("active_set[%d,%d,%d]=>comm_world 0x%lx returned.\n",
                      PE_start, logPE_stride, PE_size, (unsigned long) *comm);
        return;
    }

    /* Fast path: return a cached comm if found */
    if (coll_find_comm_cache(PE_start, logPE_stride, PE_size, comm, &strided_group)) {
        OSHMPI_DBGMSG("active_set[%d,%d,%d]=>cached comm 0x%lx returned.\n",
                      PE_start, logPE_stride, PE_size, (unsigned long) *comm);
        return;
    }

    /* Slow path: create a new communicator and cache it */

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
    OSHMPIU_free(pe_list);

    coll_set_comm_cache(PE_start, logPE_stride, PE_size, *comm, strided_group);
    OSHMPI_DBGMSG("new active_set[%d,%d,%d]=>comm 0x%lx group 0x%lx created and cached.\n",
                  PE_start, logPE_stride, PE_size, (unsigned long) *comm,
                  (unsigned long) strided_group);
}

/* Block until all PEs arrive at the barrier and all local updates
 * and remote memory updates on the default context are completed. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_barrier_all(void)
{
    /* Ensure completion of all outstanding Put, AMO, and nonblocking Put */
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_heap_win));
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_data_win));

    /* Ensure special AMO completion (e.g., AM AMOs) */
    OSHMPI_amo_flush_all(SHMEM_CTX_DEFAULT);

    /* Ensure completion of memory store */
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_heap_win));
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_data_win));

    OSHMPI_CALLMPI(MPI_Barrier(OSHMPI_global.comm_world));
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_barrier(int PE_start, int logPE_stride, int PE_size)
{
    MPI_Comm comm = MPI_COMM_NULL;

    /* Ensure completion of all outstanding Put, AMO, and nonblocking Put */
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_heap_win));
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_data_win));

    /* Ensure special AMO completion (e.g., AM AMOs) in active set */
    OSHMPI_amo_flush(SHMEM_CTX_DEFAULT, PE_start, logPE_stride, PE_size);

    /* Ensure completion of memory store */
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_heap_win));
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_data_win));

    coll_acquire_comm(PE_start, logPE_stride, PE_size, &comm);
    OSHMPI_CALLMPI(MPI_Barrier(comm));
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_sync_all(void)
{
    /* Ensure completion of previously issued memory store */
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_heap_win));
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_data_win));

    OSHMPI_CALLMPI(MPI_Barrier(OSHMPI_global.comm_world));
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_sync(int PE_start, int logPE_stride, int PE_size)
{
    MPI_Comm comm = MPI_COMM_NULL;

    /* Ensure completion of previously issued memory store */
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_heap_win));
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_data_win));

    coll_acquire_comm(PE_start, logPE_stride, PE_size, &comm);
    OSHMPI_CALLMPI(MPI_Barrier(comm));
}

/* Return 1 if root is included in the active set, otherwise 0. */
OSHMPI_STATIC_INLINE_PREFIX int coll_check_root_in_active_set(int PE_root,
                                                              int PE_start, int logPE_stride,
                                                              int PE_size)
{
    int i, included = 0;
    const int pe_stride = 1 << logPE_stride;    /* Implement 2^pe_logs with bitshift. */
    for (i = 0; i < PE_size; i++) {
        if (PE_root == PE_start + i * pe_stride) {
            included = 1;
            break;
        }
    }
    return included;
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_broadcast(void *dest, const void *source, size_t nelems,
                                                  MPI_Datatype mpi_type, int PE_root, int PE_start,
                                                  int logPE_stride, int PE_size)
{
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Aint target_disp = -1;
    MPI_Win win = MPI_WIN_NULL;

    coll_acquire_comm(PE_start, logPE_stride, PE_size, &comm);

    /* Special path: directly use MPI_Bcast if root is included in active set */
    if (coll_check_root_in_active_set(PE_root, PE_start, logPE_stride, PE_size)) {
        OSHMPI_CALLMPI(MPI_Bcast(PE_root ==
                                 OSHMPI_global.world_rank ? (void *) source : dest, nelems,
                                 mpi_type, PE_root, comm));
    } else {

        /* Generic path: every PE in active set gets data from root
         * FIXME: the semantics ensures dest is updated only on local PE at return,
         * thus we assume barrier is unneeded.*/
        OSHMPI_translate_win_and_disp(source, &win, &target_disp);
        OSHMPI_ASSERT(target_disp >= 0 && win != MPI_WIN_NULL);

        OSHMPI_CALLMPI(MPI_Get
                       (dest, nelems, mpi_type, PE_root, target_disp, nelems, mpi_type, win));
        OSHMPI_CALLMPI(MPI_Win_flush_local(PE_root, win));
    }
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_collect(void *dest, const void *source, size_t nelems,
                                                MPI_Datatype mpi_type, int PE_start,
                                                int logPE_stride, int PE_size)
{
    MPI_Comm comm = MPI_COMM_NULL;
    int *rcounts, *rdispls;
    unsigned int same_nelems = 0;

    coll_acquire_comm(PE_start, logPE_stride, PE_size, &comm);

    /* collect allows each PE to have different nelems. */
    rcounts = OSHMPIU_malloc(PE_size * sizeof(int));
    OSHMPI_ASSERT(rcounts);

    rdispls = OSHMPIU_malloc(PE_size * sizeof(int));
    OSHMPI_ASSERT(rdispls);

    OSHMPI_CALLMPI(MPI_Allgather(&nelems, 1, MPI_INT, rcounts, 1, MPI_INT, comm));

    rdispls[0] = 0;
    same_nelems = (nelems == rcounts[0]) ? 1 : 0;
    for (int i = 1; i < PE_size; i++) {
        rdispls[i] = rdispls[i - 1] + rcounts[i - 1];
        same_nelems &= (nelems == rcounts[i]);
    }

    if (same_nelems)    /* call faster allgather if same nelems on all PEs */
        OSHMPI_CALLMPI(MPI_Allgather(source, nelems, mpi_type, dest, nelems, mpi_type, comm));
    else
        OSHMPI_CALLMPI(MPI_Allgatherv(source, nelems, mpi_type, dest, rcounts, rdispls,
                                      mpi_type, comm));

    OSHMPIU_free(rdispls);
    OSHMPIU_free(rcounts);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_fcollect(void *dest, const void *source, size_t nelems,
                                                 MPI_Datatype mpi_type, int PE_start,
                                                 int logPE_stride, int PE_size)
{
    MPI_Comm comm = MPI_COMM_NULL;

    coll_acquire_comm(PE_start, logPE_stride, PE_size, &comm);

    OSHMPI_CALLMPI(MPI_Allgather(source, nelems, mpi_type, dest, nelems, mpi_type, comm));
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_alltoall(void *dest, const void *source, size_t nelems,
                                                 MPI_Datatype mpi_type, int PE_start,
                                                 int logPE_stride, int PE_size)
{
    MPI_Comm comm = MPI_COMM_NULL;

    coll_acquire_comm(PE_start, logPE_stride, PE_size, &comm);

    OSHMPI_CALLMPI(MPI_Alltoall(source, nelems, mpi_type, dest, nelems, mpi_type, comm));
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_alltoalls(void *dest, const void *source, ptrdiff_t dst,
                                                  ptrdiff_t sst, size_t nelems,
                                                  MPI_Datatype mpi_type, int PE_start,
                                                  int logPE_stride, int PE_size)
{
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Datatype sdtype = MPI_DATATYPE_NULL, rdtype = MPI_DATATYPE_NULL;
    size_t scount, rcount;

    /* Values of dst, sst, nelems must be equal on all PEs. When dst=sst=1, same as alltoall.
     * TODO: not sure if alltoall with ddt is faster or alltoallv is faster */

    /* Create derived datatypes if strided > 1, otherwise directly use basic datatype;
     * when dst == sst, reuse send datatype. */
    OSHMPI_create_strided_dtype(nelems, sst, mpi_type, nelems * sst /* required extent */ ,
                                &scount, &sdtype);
    if (dst == sst) {
        rdtype = sdtype;
        rcount = scount;
    } else
        OSHMPI_create_strided_dtype(nelems, dst, mpi_type, nelems * dst /* required extent */ ,
                                    &rcount, &rdtype);

    /* TODO: check non-int inputs exceeds int limit */

    coll_acquire_comm(PE_start, logPE_stride, PE_size, &comm);

    OSHMPI_CALLMPI(MPI_Alltoall(source, (int) scount, sdtype, dest, (int) rcount, rdtype, comm));

    OSHMPI_free_strided_dtype(mpi_type, &sdtype);
    if (dst != sst)
        OSHMPI_free_strided_dtype(mpi_type, &rdtype);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_allreduce(void *dest, const void *source, int count,
                                                  MPI_Datatype mpi_type, MPI_Op op, int PE_start,
                                                  int logPE_stride, int PE_size)
{
    MPI_Comm comm = MPI_COMM_NULL;

    coll_acquire_comm(PE_start, logPE_stride, PE_size, &comm);

    /* source and dest may be the same array, but may not be overlapping. */
    OSHMPI_CALLMPI(MPI_Allreduce((source == dest) ? MPI_IN_PLACE : source,
                                 dest, count, mpi_type, op, comm));
}

#endif /* INTERNAL_COLL_IMPL_H */

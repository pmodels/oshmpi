/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_COLL_IMPL_H
#define INTERNAL_COLL_IMPL_H

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_coll_initialize(void)
{
    OSHMPI_global.comm_cache_list.nobjs = 0;
    OSHMPI_global.comm_cache_list.head = NULL;
    OSHMPI_THREAD_INIT_CS(&OSHMPI_global.comm_cache_list_cs);

    /* FIXME: do we need preallocated cache pool allocated by the main
     * thread ? The cache object may be created by any of the threads
     * in multithreaded program. */
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_coll_finalize(void)
{
    OSHMPI_comm_cache_obj_t *cobj, *tmp;

    /* Release all cached comm */
    LL_FOREACH_SAFE(OSHMPI_global.comm_cache_list.head, cobj, tmp) {
        LL_DELETE(OSHMPI_global.comm_cache_list.head, cobj);
        OSHMPI_CALLMPI(MPI_Group_free(&cobj->group));
        OSHMPI_CALLMPI(MPI_Comm_free(&cobj->comm));
        OSHMPIU_free(cobj);
        OSHMPI_global.comm_cache_list.nobjs--;
    }
    OSHMPI_ASSERT(OSHMPI_global.comm_cache_list.nobjs == 0);
    OSHMPI_THREAD_DESTROY_CS(&OSHMPI_global.comm_cache_list_cs);
}

/* Cache a newly created comm.
 * Note that we have to cache all comms to ensure it is cached on all involved pes.
 * However, we expect that the amount of different active sets will be small.*/
OSHMPI_STATIC_INLINE_PREFIX void coll_set_comm_cache(int PE_start, int logPE_stride, int PE_size,
                                                     MPI_Comm comm, MPI_Group group)
{
    OSHMPI_comm_cache_obj_t *cobj = NULL;

    cobj = OSHMPIU_malloc(sizeof(OSHMPI_comm_cache_obj_t));
    OSHMPI_ASSERT(cobj);

    /* Set new comm */
    cobj->pe_start = PE_start;
    cobj->pe_stride = logPE_stride;
    cobj->pe_size = PE_size;
    cobj->comm = comm;
    cobj->group = group;

    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.comm_cache_list_cs);
    /* Insert in head, O(1) */
    LL_PREPEND(OSHMPI_global.comm_cache_list.head, cobj);
    OSHMPI_global.comm_cache_list.nobjs++;
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.comm_cache_list_cs);
}

/* Find if cached comm already exists. */
OSHMPI_STATIC_INLINE_PREFIX int coll_find_comm_cache(int PE_start, int logPE_stride, int PE_size,
                                                     MPI_Comm * comm, MPI_Group * group)
{
    int found = 0;
    OSHMPI_comm_cache_obj_t *cobj = NULL;

    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.comm_cache_list_cs);
    cobj = OSHMPI_global.comm_cache_list.head;
    LL_FOREACH(OSHMPI_global.comm_cache_list.head, cobj) {
        if (cobj->pe_start == PE_start && cobj->pe_stride == logPE_stride
            && cobj->pe_size == PE_size) {
            found = 1;
            *comm = cobj->comm;
            *group = cobj->group;
            break;
        }
    }
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.comm_cache_list_cs);
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
#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_ictx.win));
#else
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_heap_ictx.win));
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_data_ictx.win));
#endif
    /* Ensure AM completion (e.g., AM AMOs) */
    OSHMPI_am_flush_all(SHMEM_CTX_DEFAULT);

    /* Ensure completion of memory store */
#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_ictx.win));
#else
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_heap_ictx.win));
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_data_ictx.win));
#endif
    OSHMPI_am_progress_mpi_barrier(OSHMPI_global.comm_world);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_barrier(int PE_start, int logPE_stride, int PE_size)
{
    MPI_Comm comm = MPI_COMM_NULL;

    /* Ensure completion of all outstanding Put, AMO, and nonblocking Put */
#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_ictx.win));
#else
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_heap_ictx.win));
    OSHMPI_CALLMPI(MPI_Win_flush_all(OSHMPI_global.symm_data_ictx.win));
#endif

    /* Ensure AM completion (e.g., AM AMOs) */
    OSHMPI_am_flush(SHMEM_CTX_DEFAULT, PE_start, logPE_stride, PE_size);

    /* Ensure completion of memory store */
#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_ictx.win));
#else
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_heap_ictx.win));
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_data_ictx.win));
#endif

    coll_acquire_comm(PE_start, logPE_stride, PE_size, &comm);
    OSHMPI_am_progress_mpi_barrier(comm);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_sync_all(void)
{
    /* Ensure completion of previously issued memory store */
#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_ictx.win));
#else
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_heap_ictx.win));
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_data_ictx.win));
#endif
    OSHMPI_am_progress_mpi_barrier(OSHMPI_global.comm_world);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_sync(int PE_start, int logPE_stride, int PE_size)
{
    MPI_Comm comm = MPI_COMM_NULL;

    /* Ensure completion of previously issued memory store */
#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_ictx.win));
#else
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_heap_ictx.win));
    OSHMPI_CALLMPI(MPI_Win_sync(OSHMPI_global.symm_data_ictx.win));
#endif

    coll_acquire_comm(PE_start, logPE_stride, PE_size, &comm);
    OSHMPI_am_progress_mpi_barrier(comm);
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
    OSHMPI_ictx_t *ictx = NULL;

    coll_acquire_comm(PE_start, logPE_stride, PE_size, &comm);

    /* Special path: directly use MPI_Bcast if root is included in active set */
    if (coll_check_root_in_active_set(PE_root, PE_start, logPE_stride, PE_size)) {
        OSHMPI_am_progress_mpi_bcast(PE_root ==
                                     OSHMPI_global.world_rank ? (void *) source : dest, nelems,
                                     mpi_type, PE_root, comm);
    } else {

        /* Generic path: every PE in active set gets data from root
         * FIXME: the semantics ensures dest is updated only on local PE at return,
         * thus we assume barrier is unneeded.*/
        OSHMPI_translate_ictx_disp(SHMEM_CTX_DEFAULT, source, PE_root, &target_disp, &ictx,
                                   NULL /*sobj_handle_ptr */);
        OSHMPI_ASSERT(target_disp >= 0 && ictx);

        OSHMPI_CALLMPI(MPI_Get
                       (dest, nelems, mpi_type, PE_root, target_disp, nelems, mpi_type, ictx->win));
        OSHMPI_CALLMPI(MPI_Win_flush_local(PE_root, ictx->win));
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

    OSHMPI_am_progress_mpi_allgather(&nelems, 1, MPI_INT, rcounts, 1, MPI_INT, comm);

    rdispls[0] = 0;
    same_nelems = (nelems == rcounts[0]) ? 1 : 0;
    for (int i = 1; i < PE_size; i++) {
        rdispls[i] = rdispls[i - 1] + rcounts[i - 1];
        same_nelems &= (nelems == rcounts[i]);
    }

    if (same_nelems)    /* call faster allgather if same nelems on all PEs */
        OSHMPI_am_progress_mpi_allgather(source, nelems, mpi_type, dest, nelems, mpi_type, comm);
    else
        OSHMPI_am_progress_mpi_allgatherv(source, nelems, mpi_type, dest, rcounts, rdispls,
                                          mpi_type, comm);

    OSHMPIU_free(rdispls);
    OSHMPIU_free(rcounts);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_fcollect(void *dest, const void *source, size_t nelems,
                                                 MPI_Datatype mpi_type, int PE_start,
                                                 int logPE_stride, int PE_size)
{
    MPI_Comm comm = MPI_COMM_NULL;

    coll_acquire_comm(PE_start, logPE_stride, PE_size, &comm);

    OSHMPI_am_progress_mpi_allgather(source, nelems, mpi_type, dest, nelems, mpi_type, comm);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_alltoall(void *dest, const void *source, size_t nelems,
                                                 MPI_Datatype mpi_type, int PE_start,
                                                 int logPE_stride, int PE_size)
{
    MPI_Comm comm = MPI_COMM_NULL;

    coll_acquire_comm(PE_start, logPE_stride, PE_size, &comm);

    OSHMPI_am_progress_mpi_alltoall(source, nelems, mpi_type, dest, nelems, mpi_type, comm);
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

    OSHMPI_am_progress_mpi_alltoall(source, (int) scount, sdtype, dest, (int) rcount, rdtype, comm);

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
    OSHMPI_am_progress_mpi_allreduce((source == dest) ? MPI_IN_PLACE : source,
                                     dest, count, mpi_type, op, comm);
}

#endif /* INTERNAL_COLL_IMPL_H */

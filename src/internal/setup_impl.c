/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include "oshmpi_impl.h"

#if defined(USE_LINUX)
/* http://www.salbut.net/public/gcc-pdf/ld.pdf */
#include <unistd.h>
extern char __data_start;
extern char _end;
#elif defined(USE_APPLE)
http:  //www.manpagez.com/man/3/get_etext/
#include <mach-o/getsect.h>
unsigned long get_end();
unsigned long get_etext();
#endif

OSHMPI_global_t OSHMPI_global = { 0 };
OSHMPI_env_t OSHMPI_env = { 0 };

OSHMPI_STATIC_INLINE_PREFIX void initialize_symm_text(void)
{
    OSHMPI_global.symm_data_win = MPI_WIN_NULL;

#if defined(USE_LINUX)
    OSHMPI_global.symm_data_base = (void *) &__data_start;
    OSHMPI_global.symm_data_size = ((char *) &_end - (char *) &__data_start);
#elif defined(USE_APPLE)
    OSHMPI_global.symm_data_base = get_etext();
    OSHMPI_global.symm_data_size = get_end() - get_etext();
#else
    OSHMPI_ERR_ABORT("platform is not supported");
#endif

    /* Allocate RMA window */
    OSHMPI_CALLMPI(MPI_Win_create
                   (OSHMPI_global.symm_data_base, (MPI_Aint) OSHMPI_global.symm_data_size,
                    1 /* disp_unit */ , MPI_INFO_NULL, OSHMPI_global.comm_world,
                    &OSHMPI_global.symm_data_win));

    OSHMPI_CALLMPI(MPI_Win_lock_all(MPI_MODE_NOCHECK, OSHMPI_global.symm_data_win));
}

OSHMPI_STATIC_INLINE_PREFIX void initialize_symm_heap(void)
{
    uint64_t symm_heap_size;
    MPI_Info win_info = MPI_INFO_NULL;

    OSHMPI_global.symm_heap_base = NULL;
    OSHMPI_global.symm_heap_mspace = NULL;
    OSHMPI_global.symm_heap_win = MPI_WIN_NULL;
    OSHMPI_global.symm_heap_size = OSHMPI_env.symm_heap_size;

    /* Ensure extra bookkeeping space in MSPACE */
    symm_heap_size = (uint64_t) OSHMPI_global.symm_heap_size + OSHMPI_DLMALLOC_MIN_MSPACE_SIZE;

    /* Allocate RMA window */
    OSHMPI_CALLMPI(MPI_Info_create(&win_info));
    OSHMPI_CALLMPI(MPI_Info_set(win_info, "alloc_shm", "true"));

    OSHMPI_CALLMPI(MPI_Win_allocate((MPI_Aint) symm_heap_size, 1 /* disp_unit */ , win_info,
                                    OSHMPI_global.comm_world, &OSHMPI_global.symm_heap_base,
                                    &OSHMPI_global.symm_heap_win));
    OSHMPI_ASSERT(OSHMPI_global.symm_heap_base != NULL);

    /* Initialize MSPACE */
    OSHMPI_global.symm_heap_mspace = create_mspace_with_base(OSHMPI_global.symm_heap_base,
                                                             symm_heap_size,
                                                             OSHMPI_global.thread_level ==
                                                             SHMEM_THREAD_MULTIPLE ? 1 : 0);
    OSHMPI_ASSERT(OSHMPI_global.symm_heap_mspace != NULL);

    OSHMPI_CALLMPI(MPI_Win_lock_all(MPI_MODE_NOCHECK, OSHMPI_global.symm_heap_win));
    OSHMPI_CALLMPI(MPI_Info_free(&win_info));
}

OSHMPI_STATIC_INLINE_PREFIX void initialize_env(void)
{
    char *val = NULL;

    /* Number of bytes to allocate for symmetric heap. */
    OSHMPI_env.symm_heap_size = OSHMPI_DEFAULT_SYMM_HEAP_SIZE;
    val = getenv("SHMEM_SYMMETRIC_SIZE");
    if (val && strlen(val))
        OSHMPI_env.symm_heap_size = (unsigned long) atol(val);
    if (OSHMPI_env.symm_heap_size < 0)
        OSHMPI_ERR_ABORT("Invalid SHMEM_SYMMETRIC_SIZE: %ld\n", OSHMPI_env.symm_heap_size);

    /* FIXME: determine system available memory size */

    /* Debug message. Any non-zero value will enable debug. */
    OSHMPI_env.debug = 0;
    val = getenv("SHMEM_DEBUG");
    if (val && strlen(val))
        OSHMPI_env.debug = atoi(val);
    if (OSHMPI_env.debug != 0)
        OSHMPI_env.debug = 1;
}

int OSHMPI_initialize_thread(int required, int *provided)
{
    int mpi_errno = MPI_SUCCESS;
    int mpi_provided = 0;

    if (OSHMPI_global.is_initialized)
        goto fn_exit;

    if (required != SHMEM_THREAD_SINGLE && required != SHMEM_THREAD_FUNNELED
        && required != SHMEM_THREAD_SERIALIZED && required != SHMEM_THREAD_MULTIPLE)
        OSHMPI_ERR_ABORT("Unknown OpenSHMEM thread support level: %d\n", required);

    /* FIXME: we simply define the value of shmem thread levels
     * using MPI equivalents. A translation might be needed if such setting is not OK. */
    OSHMPI_CALLMPI(MPI_Init_thread(NULL, NULL, required, &mpi_provided));
    if (mpi_provided != required) {
        OSHMPI_ERR_ABORT("The MPI library does not support the required thread support:"
                         "required: %s, provided: %s.\n",
                         OSHMPI_thread_level_str(required), OSHMPI_thread_level_str(mpi_provided));
    }
    OSHMPI_global.thread_level = mpi_provided;

    /* Duplicate comm world for oshmpi use. */
    OSHMPI_CALLMPI(MPI_Comm_dup(MPI_COMM_WORLD, &OSHMPI_global.comm_world));
    OSHMPI_CALLMPI(MPI_Comm_size(OSHMPI_global.comm_world, &OSHMPI_global.world_size));
    OSHMPI_CALLMPI(MPI_Comm_rank(OSHMPI_global.comm_world, &OSHMPI_global.world_rank));
    OSHMPI_CALLMPI(MPI_Comm_group(OSHMPI_global.comm_world, &OSHMPI_global.comm_world_group));

    initialize_env();

    initialize_symm_text();

    initialize_symm_heap();

    OSHMPI_coll_initialize();

    OSHMPI_CALLMPI(MPI_Barrier(OSHMPI_global.comm_world));
    OSHMPI_global.is_initialized = 1;

  fn_exit:
    if (provided)
        *provided = OSHMPI_global.thread_level;
    return mpi_errno;
}

OSHMPI_STATIC_INLINE_PREFIX int finalize_impl(void)
{
    int mpi_errno = MPI_SUCCESS;

    OSHMPI_coll_finalize();

    /* Implicit global barrier is required to ensure
     * that pending communications are completed and that no resources
     * are released until all PEs have entered shmem_finalize.
     * The completion part is ensured in unlock calls.*/
    OSHMPI_CALLMPI(MPI_Barrier(OSHMPI_global.comm_world));

    if (OSHMPI_global.symm_heap_win != MPI_WIN_NULL) {
        OSHMPI_CALLMPI(MPI_Win_unlock_all(OSHMPI_global.symm_heap_win));
        OSHMPI_CALLMPI(MPI_Win_free(&OSHMPI_global.symm_heap_win));
    }
    if (OSHMPI_global.symm_data_win != MPI_WIN_NULL) {
        OSHMPI_CALLMPI(MPI_Win_unlock_all(OSHMPI_global.symm_data_win));
        OSHMPI_CALLMPI(MPI_Win_free(&OSHMPI_global.symm_data_win));
    }

    OSHMPI_global.is_initialized = 0;

    OSHMPI_CALLMPI(MPI_Group_free(&OSHMPI_global.comm_world_group));
    OSHMPI_CALLMPI(MPI_Comm_free(&OSHMPI_global.comm_world));
    OSHMPI_CALLMPI(MPI_Finalize());

    return mpi_errno;
}

/* Implicitly called at program exit, valid only when program is initialized
 * by start_pes and the finalize call is not explicitly called. */
void OSHMPI_implicit_finalize(void)
{
    if (OSHMPI_global.is_start_pes_initialized && OSHMPI_global.is_initialized)
        finalize_impl();
}

int OSHMPI_finalize(void)
{
    int mpi_errno = MPI_SUCCESS;

    /* Skip if a finalize is already called or the program is not
     * initialized yet. */
    if (OSHMPI_global.is_initialized)
        mpi_errno = finalize_impl();

    return mpi_errno;
}

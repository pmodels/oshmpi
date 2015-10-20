/* BSD-2 License.  Written by Jeff Hammond. */

#ifdef EXTENSION_ARMCI_STRIDED

#include "shmem-internals.h"

void oshmpix_put_strided_2d(MPI_Datatype mpi_type, void *target, const void *source, 
                         ptrdiff_t target_ptrdiff, ptrdiff_t source_ptrdiff, size_t len, int pe)
{
#if SHMEM_DEBUG>3
    printf("[%d] oshmpix_put_strided_2d: type=%d, target=%p, source=%p, len=%zu, pe=%d \n", 
                    shmem_world_rank, mpi_type, target, source, len, pe);
    fflush(stdout);
#endif

    int count = 0;
    if ( likely(len<(size_t)INT32_MAX) ) { /* need second check if size_t is signed */
        count = len;
    } else {
        /* TODO generate derived type ala BigMPI */
        oshmpi_abort(len%INT32_MAX, "oshmpix_put_strided_2d: count exceeds the range of a 32b integer");
    }

    enum shmem_window_id_e win_id;
    shmem_offset_t win_offset;

    if (oshmpi_window_offset(target, pe, &win_id, &win_offset)) {
        oshmpi_abort(pe, "oshmpi_window_offset failed to find iput target");
    }
#if SHMEM_DEBUG>3
    printf("[%d] win_id=%d, offset=%lld \n", 
           shmem_world_rank, win_id, (long long)win_offset);
    fflush(stdout);
#endif

    MPI_Win win = (win_id==SHMEM_SHEAP_WINDOW) ? shmem_sheap_win : shmem_etext_win;
#ifdef ENABLE_SMP_OPTIMIZATIONS
    if (0) {
        /* TODO */
    } else
#endif
    {
        assert( (ptrdiff_t)INT32_MIN<target_ptrdiff && target_ptrdiff<(ptrdiff_t)INT32_MAX );
        assert( (ptrdiff_t)INT32_MIN<source_ptrdiff && source_ptrdiff<(ptrdiff_t)INT32_MAX );

        int target_stride = (int) target_ptrdiff;
        int source_stride = (int) source_ptrdiff;

        MPI_Datatype source_type;
        MPI_Type_vector(count, 1, source_stride, mpi_type, &source_type);
        MPI_Type_commit(&source_type);

        MPI_Datatype target_type;
        if (target_stride!=source_stride) {
            MPI_Type_vector(count, 1, target_stride, mpi_type, &target_type);
            MPI_Type_commit(&target_type);
        } else {
            target_type = source_type;
        }

#ifdef ENABLE_RMA_ORDERING
        /* ENABLE_RMA_ORDERING means "RMA operations are ordered" */
        MPI_Accumulate(source, 1, source_type,                   /* origin */
                       pe, (MPI_Aint)win_offset, 1, target_type, /* target */
                       MPI_REPLACE,                              /* atomic, ordered Put */
                       win);
#else
        MPI_Put(source, 1, source_type,                   /* origin */
                pe, (MPI_Aint)win_offset, 1, target_type, /* target */
                win);
#endif
        MPI_Win_flush_local(pe, win);

        if (target_stride!=source_stride) {
            MPI_Type_free(&target_type);
        }
        MPI_Type_free(&source_type);
    }

    return;
}

void oshmpix_get_strided_2d(MPI_Datatype mpi_type, void *target, const void *source, 
                         ptrdiff_t target_ptrdiff, ptrdiff_t source_ptrdiff, size_t len, int pe)
{
#if SHMEM_DEBUG>3
    printf("[%d] oshmpix_get_strided_2d: type=%d, target=%p, source=%p, len=%zu, pe=%d \n", 
                    shmem_world_rank, mpi_type, target, source, len, pe);
    fflush(stdout);
#endif

    int count = 0;
    if ( likely(len<(size_t)INT32_MAX) ) { /* need second check if size_t is signed */
        count = len;
    } else {
        /* TODO generate derived type ala BigMPI */
        oshmpi_abort(len%INT32_MAX, "oshmpix_get_strided_2d: count exceeds the range of a 32b integer");
    }

    enum shmem_window_id_e win_id;
    shmem_offset_t win_offset;

    if (oshmpi_window_offset(source, pe, &win_id, &win_offset)) {
        oshmpi_abort(pe, "oshmpi_window_offset failed to find iget source");
    }
#if SHMEM_DEBUG>3
    printf("[%d] win_id=%d, offset=%lld \n", 
           shmem_world_rank, win_id, (long long)win_offset);
    fflush(stdout);
#endif

    MPI_Win win = (win_id==SHMEM_SHEAP_WINDOW) ? shmem_sheap_win : shmem_etext_win;
#ifdef ENABLE_SMP_OPTIMIZATIONS
    if (0) {
        /* TODO */
    } else 
#endif
    {
        assert( (ptrdiff_t)INT32_MIN<target_ptrdiff && target_ptrdiff<(ptrdiff_t)INT32_MAX );
        assert( (ptrdiff_t)INT32_MIN<source_ptrdiff && source_ptrdiff<(ptrdiff_t)INT32_MAX );

        int target_stride = (int) target_ptrdiff;
        int source_stride = (int) source_ptrdiff;

        MPI_Datatype source_type;
        MPI_Type_vector(count, 1, source_stride, mpi_type, &source_type);
        MPI_Type_commit(&source_type);

        MPI_Datatype target_type;
        if (target_stride!=source_stride) {
            MPI_Type_vector(count, 1, target_stride, mpi_type, &target_type);
            MPI_Type_commit(&target_type);
        } else {
            target_type = source_type;
        }

#ifdef ENABLE_RMA_ORDERING
        /* ENABLE_RMA_ORDERING means "RMA operations are ordered" */
        MPI_Get_accumulate(NULL, 0, MPI_DATATYPE_NULL,                   /* origin */
                           target, 1, target_type,                   /* result */
                           pe, (MPI_Aint)win_offset, 1, source_type, /* remote */
                           MPI_NO_OP,                                    /* atomic, ordered Get */
                           win);
#else
        MPI_Get(target, 1, target_type,                   /* result */
                pe, (MPI_Aint)win_offset, 1, source_type, /* remote */
                win);
#endif
        MPI_Win_flush_local(pe, win);

        if (target_stride!=source_stride) 
            MPI_Type_free(&target_type);
        MPI_Type_free(&source_type);
    }

    return;
}

#endif

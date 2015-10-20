#ifndef TYPE_CONTIGUOUS_X_H
#define TYPE_CONTIGUOUS_X_H

/* This code was taken from the BigMPI project.
 * Please see https://github.com/jeffhammond/BigMPI
 * for all technical details. */

#include <limits.h>
/*
 * Synopsis
 *
 * int MPIX_Type_contiguous_x(size_t         count,
 *                            MPI_Datatype   oldtype,
 *                            MPI_Datatype * newtype)
 *
 *  Input Parameters
 *
 *   count             replication count (nonnegative integer)
 *   oldtype           old datatype (handle)
 *
 * Output Parameter
 *
 *   newtype           new datatype (handle)
 *
 */
static int MPIX_Type_contiguous_x(size_t count, MPI_Datatype oldtype, MPI_Datatype * newtype)
{
    int c = count/(size_t)INT_MAX;
    int r = count%(size_t)INT_MAX;

    MPI_Datatype chunks;
    MPI_Type_vector(c, INT_MAX, INT_MAX, oldtype, &chunks);

    MPI_Datatype remainder;
    MPI_Type_contiguous(r, oldtype, &remainder);

    MPI_Aint lb /* unused */, extent;
    MPI_Type_get_extent(oldtype, &lb, &extent);

    MPI_Aint remdisp          = (MPI_Aint)c*(size_t)INT_MAX*extent;
    int blocklengths[2]       = {1,1};
    MPI_Aint displacements[2] = {0,remdisp};
    MPI_Datatype types[2]     = {chunks,remainder};
    MPI_Type_create_struct(2, blocklengths, displacements, types, newtype);

    MPI_Type_free(&chunks);
    MPI_Type_free(&remainder);

    return MPI_SUCCESS;
}

#endif // TYPE_CONTIGUOUS_X_H

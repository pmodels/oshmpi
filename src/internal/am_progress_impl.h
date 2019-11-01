/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_AM_PROGRESS_IMPL_H
#define INTERNAL_AM_PROGRESS_IMPL_H

#include "oshmpi_impl.h"

/* Define wrapper of blocking MPI calls used during SHMEM program.
 * If OSHMPI_ENABLE_ASYNC_THREAD is set, the MPI blocking version is called;
 * otherwise the MPI nonblocking version is called with SHMEM active message
 * progress manual polling. */

#define OSHMPI_AM_PROGRESS_MPI(req, stat) do {                                 \
    int am_mpi_flag = 0;                                                       \
    while (1) {                                                                \
        OSHMPI_CALLMPI(MPI_Test(&req, &am_mpi_flag, stat));                    \
        if (am_mpi_flag) break; /* skip AM progress if complete immediately */ \
        OSHMPI_progress(OSHMPI_PROGRESS_NONBLOCKING, NULL);                    \
    }                                                                          \
} while (0)

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_progress_poll_am(void)
{
#if defined(OSHMPI_ENABLE_ASYNC_THREAD) || defined(OSHMPI_RUNTIME_ASYNC_THREAD)
    if (OSHMPI_env.enable_async_thread) {
        return;
    }
#endif
    OSHMPI_progress(OSHMPI_PROGRESS_NONBLOCKING, NULL);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_send(const void *buf, int count,
                                                             MPI_Datatype datatype, int dest,
                                                             int tag, MPI_Comm comm)
{
#ifdef OSHMPI_ENABLE_ASYNC_THREAD
    OSHMPI_CALLMPI(MPI_Send(buf, count, datatype, dest, tag, comm));
    return;
#elif defined(OSHMPI_RUNTIME_ASYNC_THREAD)
    if (OSHMPI_env.enable_async_thread) {
        OSHMPI_CALLMPI(MPI_Send(buf, count, datatype, dest, tag, comm));
        return;
    } else
#endif
    {
        MPI_Request am_mpi_req = MPI_REQUEST_NULL;
        OSHMPI_CALLMPI(MPI_Isend(buf, count, datatype, dest, tag, comm, &am_mpi_req));
        OSHMPI_AM_PROGRESS_MPI(am_mpi_req, MPI_STATUS_IGNORE);
    }
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_recv(void *buf, int count,
                                                             MPI_Datatype datatype, int src,
                                                             int tag, MPI_Comm comm,
                                                             MPI_Status * status)
{
#ifdef OSHMPI_ENABLE_ASYNC_THREAD
    OSHMPI_CALLMPI(MPI_Recv(buf, count, datatype, src, tag, comm, status));
    return;
#elif defined(OSHMPI_RUNTIME_ASYNC_THREAD)
    if (OSHMPI_env.enable_async_thread) {
        OSHMPI_CALLMPI(MPI_Recv(buf, count, datatype, src, tag, comm, status));
        return;
    } else
#endif
    {
        MPI_Request am_mpi_req = MPI_REQUEST_NULL;
        OSHMPI_CALLMPI(MPI_Irecv(buf, count, datatype, src, tag, comm, &am_mpi_req));
        OSHMPI_AM_PROGRESS_MPI(am_mpi_req, status);
    }
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_waitall(int count,
                                                                MPI_Request array_of_requests[],
                                                                MPI_Status array_of_statuses[])
{
#ifdef OSHMPI_ENABLE_ASYNC_THREAD
    OSHMPI_CALLMPI(MPI_Waitall(count, array_of_requests, array_of_statuses));
    return;
#elif defined(OSHMPI_RUNTIME_ASYNC_THREAD)
    if (OSHMPI_env.enable_async_thread) {
        OSHMPI_CALLMPI(MPI_Waitall(count, array_of_requests, array_of_statuses));
        return;
    } else
#endif
    {
        int am_mpi_flag = 0;
        while (1) {
            OSHMPI_CALLMPI(MPI_Testall(count, array_of_requests, &am_mpi_flag, array_of_statuses));
            if (am_mpi_flag)    /* skip AM progress if complete immediately */
                break;
            OSHMPI_progress(OSHMPI_PROGRESS_NONBLOCKING, NULL);
        }
    }
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_barrier(MPI_Comm comm)
{
#ifdef OSHMPI_ENABLE_ASYNC_THREAD
    OSHMPI_CALLMPI(MPI_Barrier(comm));
    return;
#elif defined(OSHMPI_RUNTIME_ASYNC_THREAD)
    if (OSHMPI_env.enable_async_thread) {
        OSHMPI_CALLMPI(MPI_Barrier(comm));
        return;
    } else
#endif
    {
        MPI_Request am_mpi_req = MPI_REQUEST_NULL;
        OSHMPI_CALLMPI(MPI_Ibarrier(comm, &am_mpi_req));
        OSHMPI_AM_PROGRESS_MPI(am_mpi_req, MPI_STATUS_IGNORE);
    }
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_bcast(void *buffer, int count,
                                                              MPI_Datatype datatype, int root,
                                                              MPI_Comm comm)
{
#ifdef OSHMPI_ENABLE_ASYNC_THREAD
    OSHMPI_CALLMPI(MPI_Bcast(buffer, count, datatype, root, comm));
    return;
#elif defined(OSHMPI_RUNTIME_ASYNC_THREAD)
    if (OSHMPI_env.enable_async_thread) {
        OSHMPI_CALLMPI(MPI_Bcast(buffer, count, datatype, root, comm));
        return;
    } else
#endif
    {
        MPI_Request am_mpi_req = MPI_REQUEST_NULL;
        OSHMPI_CALLMPI(MPI_Ibcast(buffer, count, datatype, root, comm, &am_mpi_req));
        OSHMPI_AM_PROGRESS_MPI(am_mpi_req, MPI_STATUS_IGNORE);
    }
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_allgather(const void *sendbuf,
                                                                  int sendcount,
                                                                  MPI_Datatype sendtype,
                                                                  void *recvbuf, int recvcount,
                                                                  MPI_Datatype recvtype,
                                                                  MPI_Comm comm)
{
#ifdef OSHMPI_ENABLE_ASYNC_THREAD
    OSHMPI_CALLMPI(MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm));
    return;
#elif defined(OSHMPI_RUNTIME_ASYNC_THREAD)
    if (OSHMPI_env.enable_async_thread) {
        OSHMPI_CALLMPI(MPI_Allgather
                       (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm));
        return;
    } else
#endif
    {
        MPI_Request am_mpi_req = MPI_REQUEST_NULL;
        OSHMPI_CALLMPI(MPI_Iallgather
                       (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
                        &am_mpi_req));
        OSHMPI_AM_PROGRESS_MPI(am_mpi_req, MPI_STATUS_IGNORE);
    }
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_allgatherv(const void *sendbuf,
                                                                   int sendcount,
                                                                   MPI_Datatype sendtype,
                                                                   void *recvbuf,
                                                                   const int *recvcounts,
                                                                   const int *displs,
                                                                   MPI_Datatype recvtype,
                                                                   MPI_Comm comm)
{
#ifdef OSHMPI_ENABLE_ASYNC_THREAD
    OSHMPI_CALLMPI(MPI_Allgatherv
                   (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm));
    return;
#elif defined(OSHMPI_RUNTIME_ASYNC_THREAD)
    if (OSHMPI_env.enable_async_thread) {
        OSHMPI_CALLMPI(MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
                                      recvtype, comm));
        return;
    } else
#endif
    {
        MPI_Request am_mpi_req = MPI_REQUEST_NULL;
        OSHMPI_CALLMPI(MPI_Iallgatherv
                       (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
                        &am_mpi_req));
        OSHMPI_AM_PROGRESS_MPI(am_mpi_req, MPI_STATUS_IGNORE);
    }
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_alltoall(const void *sendbuf,
                                                                 int sendcount,
                                                                 MPI_Datatype sendtype,
                                                                 void *recvbuf, int recvcount,
                                                                 MPI_Datatype recvtype,
                                                                 MPI_Comm comm)
{
#ifdef OSHMPI_ENABLE_ASYNC_THREAD
    OSHMPI_CALLMPI(MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm));
    return;
#elif defined(OSHMPI_RUNTIME_ASYNC_THREAD)
    if (OSHMPI_env.enable_async_thread) {
        OSHMPI_CALLMPI(MPI_Alltoall
                       (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm));
        return;
    } else
#endif
    {
        MPI_Request am_mpi_req = MPI_REQUEST_NULL;
        OSHMPI_CALLMPI(MPI_Ialltoall
                       (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
                        &am_mpi_req));
        OSHMPI_AM_PROGRESS_MPI(am_mpi_req, MPI_STATUS_IGNORE);
    }
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_am_progress_mpi_allreduce(const void *sendbuf,
                                                                  void *recvbuf, int count,
                                                                  MPI_Datatype datatype,
                                                                  MPI_Op op, MPI_Comm comm)
{
#ifdef OSHMPI_ENABLE_ASYNC_THREAD
    OSHMPI_CALLMPI(MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm));
    return;
#elif defined(OSHMPI_RUNTIME_ASYNC_THREAD)
    if (OSHMPI_env.enable_async_thread) {
        OSHMPI_CALLMPI(MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm));
        return;
    } else
#endif
    {
        MPI_Request am_mpi_req = MPI_REQUEST_NULL;
        OSHMPI_CALLMPI(MPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, &am_mpi_req));
        OSHMPI_AM_PROGRESS_MPI(am_mpi_req, MPI_STATUS_IGNORE);
    }
}

#endif /* INTERNAL_AM_PROGRESS_IMPL_H */

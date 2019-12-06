#
# Copyright (C) 2018. See COPYRIGHT in top-level directory.
#

check_PROGRAMS +=  \
    hybrid_mpi/mapping_id            \
    hybrid_mpi/mapping_id_shmem_comm \
    hybrid_mpi/query_mpi_progress    \
    hybrid_mpi/threaded              \
    hybrid_mpi/init_shmem_init_shmem_finalize       \
    hybrid_mpi/init_shmem_init_mpi_finalize         \
    hybrid_mpi/init_mpi_init_mpi_finalize           \
    hybrid_mpi/init_mpi_init_shmem_finalize

# Default SHMEM init first SHMEM finalize first
hybrid_mpi_init_shmem_init_shmem_finalize_SOURCES   = hybrid_mpi/init.c
hybrid_mpi_init_shmem_init_shmem_finalize_CPPFLAGS  = $(AM_CPPFLAGS)

# Default SHMEM init first
hybrid_mpi_init_shmem_init_mpi_finalize_SOURCES   = hybrid_mpi/init.c
hybrid_mpi_init_shmem_init_mpi_finalize_CPPFLAGS  = -DTEST_MPI_FINALIZE_FIRST $(AM_CPPFLAGS)

hybrid_mpi_init_mpi_init_mpi_finalize_SOURCES   = hybrid_mpi/init.c
hybrid_mpi_init_mpi_init_mpi_finalize_CPPFLAGS  = -DTEST_MPI_INIT_FIRST -DTEST_MPI_FINALIZE_FIRST $(AM_CPPFLAGS)

# Default SHMEM finalize first
hybrid_mpi_init_mpi_init_shmem_finalize_SOURCES   = hybrid_mpi/init.c
hybrid_mpi_init_mpi_init_shmem_finalize_CPPFLAGS  = -DTEST_MPI_INIT_FIRST $(AM_CPPFLAGS)

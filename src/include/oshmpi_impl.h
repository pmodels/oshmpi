/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef OSHMPI_IMPL_H
#define OSHMPI_IMPL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <oshmpiconf.h>
#include <shmem.h>

#include "dlmalloc.h"
#include "oshmpi_util.h"

#define OSHMPI_DEFAULT_SYMM_HEAP_SIZE (1L<<31)  /* 2GB */
#define OSHMPI_DEFAULT_DEBUG 0

/* DLMALLOC minimum allocated size (see create_mspace_with_base)
 * Part (less than 128*sizeof(size_t) bytes) of this space is used for bookkeeping,
 * so the capacity must be at least this large */
#define OSHMPI_DLMALLOC_MIN_MSPACE_SIZE (128 * sizeof(size_t))

typedef struct {
    int is_initialized;
    int is_start_pes_initialized;
    int world_rank;
    int world_size;
    int thread_level;

    MPI_Comm comm_world;        /* duplicate of COMM_WORLD */

    MPI_Win symm_heap_win;
    void *symm_heap_base;
    MPI_Aint symm_heap_size;
    mspace symm_heap_mspace;
} OSHMPI_global_t;

typedef struct {
    MPI_Aint symm_heap_size;    /* SHMEM_SYMMETRIC_SIZE */
    int debug;                  /*SHMEM_DEBUG, value: 0|1 */
} OSHMPI_env_t;

extern OSHMPI_global_t OSHMPI_global;
extern OSHMPI_env_t OSHMPI_env;

int OSHMPI_initialize_thread(int required, int *provided);
void OSHMPI_implicit_finalize(void);
int OSHMPI_finalize(void);

static inline void *OSHMPI_malloc(size_t size);
static inline void OSHMPI_free(void *ptr);
static inline void *OSHMPI_realloc(void *ptr, size_t size);
static inline void *OSHMPI_align(size_t alignment, size_t size);

static inline void OSHMPI_ctx_put_nbi(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                      MPI_Datatype mpi_type, const void *origin_addr,
                                      void *target_addr, size_t nelems, int pe);
static inline void OSHMPI_ctx_put(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                  MPI_Datatype mpi_type, const void *origin_addr,
                                  void *target_addr, size_t nelems, int pe);
static inline void OSHMPI_ctx_get_nbi(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                      MPI_Datatype mpi_type, void *origin_addr,
                                      const void *target_addr, size_t nelems, int pe);
static inline void OSHMPI_ctx_get(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                  MPI_Datatype mpi_type, void *origin_addr,
                                  const void *target_addr, size_t nelems, int pe);

static inline void OSHMPI_barrier_all(void);

#include "mem_impl.h"
#include "coll_impl.h"
#include "rma_impl.h"

#endif /* OSHMPI_IMPL_H */

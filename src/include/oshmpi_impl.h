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

typedef struct OSHMPI_comm_cache_obj {
    int pe_start;
    int pe_stride;
    int pe_size;
    MPI_Comm comm;
    struct OSHMPI_comm_cache_obj *next;
} OSHMPI_comm_cache_obj_t;

typedef struct OSHMPI_comm_cache_list {
    OSHMPI_comm_cache_obj_t *head;
    int nobjs;
} OSHMPI_comm_cache_list_t;

typedef struct {
    int is_initialized;
    int is_start_pes_initialized;
    int world_rank;
    int world_size;
    int thread_level;

    MPI_Comm comm_world;        /* duplicate of COMM_WORLD */
    MPI_Group comm_world_group;

    MPI_Win symm_heap_win;
    void *symm_heap_base;
    MPI_Aint symm_heap_size;
    mspace symm_heap_mspace;

    OSHMPI_comm_cache_list_t comm_cache_list;
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
static inline void OSHMPI_ctx_iput(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                   MPI_Datatype mpi_type, const void *origin_addr,
                                   void *target_addr, ptrdiff_t dst, ptrdiff_t sst,
                                   size_t nelems, int pe);
static inline void OSHMPI_ctx_get_nbi(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                      MPI_Datatype mpi_type, void *origin_addr,
                                      const void *target_addr, size_t nelems, int pe);
static inline void OSHMPI_ctx_get(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                  MPI_Datatype mpi_type, void *origin_addr,
                                  const void *target_addr, size_t nelems, int pe);
static inline void OSHMPI_ctx_iget(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                   MPI_Datatype mpi_type, void *origin_addr,
                                   const void *target_addr, ptrdiff_t dst, ptrdiff_t sst,
                                   size_t nelems, int pe);

static inline void OSHMPI_ctx_compare_swap(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                           MPI_Datatype mpi_type, const void *origin_addr,
                                           const void *compare_addr, void *result_addr,
                                           void *target_addr, int pe);
static inline void OSHMPI_ctx_fetch_add(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                        MPI_Datatype mpi_type, const void *origin_addr,
                                        void *result_addr, void *target_addr, int pe);
static inline void OSHMPI_ctx_add(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                  MPI_Datatype mpi_type, const void *origin_addr,
                                  void *target_addr, int pe);
static inline void OSHMPI_ctx_fetch(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                    MPI_Datatype mpi_type, void *result_addr,
                                    const void *target_addr, int pe);
static inline void OSHMPI_ctx_set(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)), MPI_Datatype mpi_type,
                                  const void *origin_addr, void *target_addr, int pe);
static inline void OSHMPI_ctx_swap(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                   MPI_Datatype mpi_type, const void *origin_addr,
                                   void *result_addr, void *target_addr, int pe);
static inline void OSHMPI_ctx_fetch_and(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                        MPI_Datatype mpi_type, const void *origin_addr,
                                        void *result_addr, void *target_addr, int pe);
static inline void OSHMPI_ctx_fetch_or(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                       MPI_Datatype mpi_type, const void *origin_addr,
                                       void *result_addr, void *target_addr, int pe);
static inline void OSHMPI_ctx_and(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)), MPI_Datatype mpi_type,
                                  const void *origin_addr, void *target_addr, int pe);
static inline void OSHMPI_ctx_or(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)), MPI_Datatype mpi_type,
                                 const void *origin_addr, void *target_addr, int pe);
static inline void OSHMPI_coll_initialize(void);
static inline void OSHMPI_coll_finalize(void);
static inline void OSHMPI_barrier_all(void);

static inline void OSHMPI_ctx_fence(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)));
static inline void OSHMPI_ctx_quiet(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)));

/* Common routines for internal use */
static inline void OSHMPI_translate_win_and_disp(const void *abs_addr, MPI_Win * win_ptr,
                                                 MPI_Aint * disp_ptr)
{
    /* heap */
    if (OSHMPI_global.symm_heap_base <= abs_addr &&
        (MPI_Aint) abs_addr <= (MPI_Aint) OSHMPI_global.symm_heap_base +
        OSHMPI_global.symm_heap_size) {
        *disp_ptr = (MPI_Aint) OSHMPI_global.symm_heap_base - (MPI_Aint) abs_addr;
        *win_ptr = OSHMPI_global.symm_heap_win;
    }
}

static inline void ctx_local_complete_impl(shmem_ctx_t ctx OSHMPI_ATTRIBUTE((unused)),
                                           int pe, MPI_Win win)
{
    OSHMPI_CALLMPI(MPI_Win_flush_local(pe, win));
}

#include "mem_impl.h"
#include "coll_impl.h"
#include "rma_impl.h"
#include "amo_impl.h"
#include "order_impl.h"

#endif /* OSHMPI_IMPL_H */

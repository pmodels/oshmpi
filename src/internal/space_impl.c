/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include <shmemx.h>
#include "oshmpi_impl.h"
#ifdef OSHMPI_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

static void space_ictx_create(void *base, MPI_Aint size, MPI_Info info, OSHMPI_ictx_t * ictx)
{
    OSHMPI_CALLMPI(MPI_Win_create(base, size, 1 /* disp_unit */ , info,
                                  OSHMPI_global.comm_world, &ictx->win));
    OSHMPI_CALLMPI(MPI_Win_lock_all(MPI_MODE_NOCHECK, ictx->win));
    ictx->outstanding_op = 0;
}

static void space_ictx_destroy(OSHMPI_ictx_t * ictx)
{
    OSHMPI_ASSERT(ictx->win != MPI_WIN_NULL);
    OSHMPI_CALLMPI(MPI_Win_unlock_all(ictx->win));
    OSHMPI_CALLMPI(MPI_Win_free(&ictx->win));
    ictx->win = MPI_WIN_NULL;
}

static const char *space_memkind_str(shmemx_memkind_t memkind)
{
    switch (memkind) {
        case SHMEMX_MEM_CUDA:
            return "cuda";
            break;
        case SHMEMX_MEM_HOST:
        default:
            return "host";
            break;
    }
}

#ifdef OSHMPI_ENABLE_CUDA
static void space_cuda_mem_create(OSHMPI_space_t * space)
{
    OSHMPI_CALLCUDA(cudaMalloc(&space->heap_base, space->heap_sz));
}

static void space_cuda_mem_destroy(OSHMPI_space_t * space)
{
    OSHMPI_CALLCUDA(cudaFree(space->heap_base));
}
#else
static void space_cuda_mem_create(OSHMPI_space_t * space)
{
    OSHMPI_ERR_ABORT("Memory kind CUDA is disabled. Recompile with --enable-cuda to enable\n");
}

static void space_cuda_mem_destroy(OSHMPI_space_t * space)
{
    OSHMPI_ERR_ABORT("Memory kind CUDA is disabled. Recompile with --enable-cuda to enable\n");
}
#endif

void OSHMPI_space_initialize(void)
{
    OSHMPI_THREAD_INIT_CS(&OSHMPI_global.space_list.cs);
}

void OSHMPI_space_finalize(void)
{
    OSHMPI_THREAD_DESTROY_CS(&OSHMPI_global.space_list.cs);
}

/* Locally create a space with a symmetric heap allocated based on the user
 * specified configure options. */
void OSHMPI_space_create(shmemx_space_config_t space_config, OSHMPI_space_t ** space_ptr)
{
    OSHMPI_space_t *space = OSHMPIU_malloc(sizeof(OSHMPI_space_t));
    OSHMPI_ASSERT(space);
    OSHMPI_ASSERT(space_config.num_contexts >= 0);
    OSHMPI_ASSERT(space_config.sheap_size > 0);

    /* Allocate internal heap. Note that heap may be allocated on device.
     * Thus, we need allocate heap and the space object separately. */
    space->heap_sz = OSHMPI_ALIGN(space_config.sheap_size, OSHMPI_global.page_sz);
    OSHMPI_SET_SOBJ_HANDLE(space->sobj_handle, OSHMPI_SOBJ_SPACE_HEAP, 0);

    switch (space_config.memkind) {
        case SHMEMX_MEM_CUDA:
            space_cuda_mem_create(space);
            break;
        case SHMEMX_MEM_HOST:
        default:
            space->heap_base = OSHMPIU_malloc(space->heap_sz);
            space_config.memkind = SHMEMX_MEM_HOST;
            break;
    }
    OSHMPI_ASSERT(space->heap_base);

    /* Initialize memory pool per space */
    OSHMPIU_mempool_init(&space->mem_pool, space->heap_base, space->heap_sz, OSHMPI_global.page_sz);
    OSHMPI_THREAD_INIT_CS(&space->mem_pool_cs);

    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.space_list.cs);
    LL_APPEND(OSHMPI_global.space_list.head, space);
    OSHMPI_global.space_list.nspaces++;
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.space_list.cs);

    space->ctx_list = NULL;
    space->default_ictx.win = MPI_WIN_NULL;
    space->default_ictx.outstanding_op = 0;
    space->config = space_config;

    OSHMPI_DBGMSG
        ("create space %p, base %p, size %ld, num_contexts=%d, memkind=%d (%s), handle 0x%x\n",
         space, space->heap_base, space->heap_sz, space->config.num_contexts, space->config.memkind,
         space_memkind_str(space->config.memkind), space->sobj_handle);

    *space_ptr = (void *) space;
}

/* Locally destroy a space and free the associated symmetric heap. */
void OSHMPI_space_destroy(OSHMPI_space_t * space)
{
    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.space_list.cs);
    LL_DELETE(OSHMPI_global.space_list.head, space);
    OSHMPI_global.space_list.nspaces--;
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.space_list.cs);

    OSHMPIU_mempool_destroy(&space->mem_pool);
    OSHMPI_THREAD_DESTROY_CS(&space->mem_pool_cs);

    switch (space->config.memkind) {
        case SHMEMX_MEM_CUDA:
            space_cuda_mem_destroy(space);
            break;
        case SHMEMX_MEM_HOST:
        default:
            OSHMPIU_free(space->heap_base);
            break;
    }

    OSHMPIU_free(space);
}

/* Locally create a context associated with the space.
 * RMA/AMO and memory synchronization calls with a space_ctx will only access to the specific space. */
int OSHMPI_space_create_ctx(OSHMPI_space_t * space, long options, OSHMPI_ctx_t ** ctx_ptr)
{
    int shmem_errno = SHMEM_SUCCESS;
    int i;

    /* Space should have already be attached or no context is required at config. */
    OSHMPI_ASSERT(space->ctx_list || space->config.num_contexts == 0);

    for (i = 0; i < space->config.num_contexts; i++) {
        if (OSHMPI_ATOMIC_FLAG_CAS(space->ctx_list[i].used_flag, 0, 1) == 0) {
            *ctx_ptr = &space->ctx_list[i];
            break;
        }
    }
    if (i >= space->config.num_contexts)        /* no available context */
        shmem_errno = SHMEM_NO_CTX;

    return shmem_errno;
}

static int space_attach_idx = 0;

/* Collectively attach the space into the default team */
void OSHMPI_space_attach(OSHMPI_space_t * space)
{
    MPI_Info info = MPI_INFO_NULL;

    /* Space should not be attached yet */
    OSHMPI_ASSERT(!space->ctx_list);

    OSHMPI_CALLMPI(MPI_Info_create(&info));
    OSHMPI_set_mpi_info_args(info);

    /* Update symm object handle
     * TODO: need collectively define index when adding teams */
    OSHMPI_SET_SOBJ_HANDLE(space->sobj_handle, OSHMPI_SOBJ_SPACE_ATTACHED_HEAP, space_attach_idx);
    space_attach_idx++;

    /* Create internal window */
    space_ictx_create(space->heap_base, (MPI_Aint) space->heap_sz, info, &space->default_ictx);
    OSHMPI_DBGMSG("space_attach space %p, default ctx: base %p, size %ld, win 0x%x\n",
                  space, space->heap_base, space->heap_sz, space->default_ictx.win);

    /* TODO: assume all processes have the same config */
    /* Create explicit-context windows */
    if (space->config.num_contexts > 0) {
        space->ctx_list =
            (OSHMPI_ctx_t *) OSHMPIU_malloc(sizeof(OSHMPI_ctx_t) * space->config.num_contexts);
        int i;
        for (i = 0; i < space->config.num_contexts; i++) {
            space_ictx_create(space->heap_base, (MPI_Aint) space->heap_sz,
                              info, &space->ctx_list[i].ictx);

            /* copy into context to avoid pointer dereference in RMA/AMO path */
            space->ctx_list[i].base = space->heap_base;
            space->ctx_list[i].size = space->heap_sz;
            space->ctx_list[i].memkind = space->config.memkind;
            space->ctx_list[i].sobj_handle = space->sobj_handle;
            OSHMPI_ATOMIC_FLAG_STORE(space->ctx_list[i].used_flag, 0);

            OSHMPI_DBGMSG("attach space %p, ctx[%d]: base %p, size %ld, win 0x%x\n",
                          space, i, space->ctx_list[i].base, space->ctx_list[i].size,
                          space->ctx_list[i].ictx.win);
        }
    }

    OSHMPI_CALLMPI(MPI_Info_free(&info));
}

/* Collectively detach the space from the default team */
void OSHMPI_space_detach(OSHMPI_space_t * space)
{
    int i;

    /* Destroy internal window */
    space_ictx_destroy(&space->default_ictx);

    /* Space should have already be attached or no context is required at config */
    OSHMPI_ASSERT((space->config.num_contexts == 0 || space->ctx_list));

    /* Destroy explicit-context windows */
    for (i = 0; i < space->config.num_contexts; i++) {
        OSHMPI_ASSERT(OSHMPI_ATOMIC_FLAG_LOAD(space->ctx_list[i].used_flag) == 0);
        space_ictx_destroy(&space->ctx_list[i].ictx);
    }
    OSHMPIU_free(space->ctx_list);
    space->ctx_list = NULL;
    OSHMPI_SET_SOBJ_HANDLE(space->sobj_handle, OSHMPI_SOBJ_SPACE_HEAP, 0);
}

/* Collectively allocate a buffer from the space */
void *OSHMPI_space_malloc(OSHMPI_space_t * space, size_t size)
{
    void *ptr = NULL;

    OSHMPI_THREAD_ENTER_CS(&space->mem_pool_cs);
    ptr = OSHMPIU_mempool_alloc(&space->mem_pool, size);
    OSHMPI_THREAD_EXIT_CS(&space->mem_pool_cs);

    OSHMPI_DBGMSG("space_malloc from space %p, size %ld -> ptr %p, disp 0x%lx\n",
                  space, size, ptr, (MPI_Aint) ptr - (MPI_Aint) space->heap_base);
    OSHMPI_barrier_all();
    return ptr;
}

/* Collectively allocate a buffer from the space with byte alignment */
void *OSHMPI_space_align(OSHMPI_space_t * space, size_t alignment, size_t size)
{
    return OSHMPI_space_malloc(space, OSHMPI_ALIGN(size, alignment));
}

/* Collectively free a buffer from the space */
void OSHMPI_space_free(OSHMPI_space_t * space, void *ptr)
{
    OSHMPI_THREAD_ENTER_CS(&space->mem_pool_cs);
    OSHMPIU_mempool_free(&space->mem_pool, ptr);
    OSHMPI_THREAD_EXIT_CS(&space->mem_pool_cs);

    OSHMPI_DBGMSG("space_free from space %p, ptr %p, disp 0x%lx\n",
                  space, ptr, (MPI_Aint) ptr - (MPI_Aint) space->heap_base);
}

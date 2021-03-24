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
#ifdef OSHMPI_ENABLE_ZE
#include <level_zero/ze_api.h>
#endif

static void space_ictx_create(void *base, MPI_Aint size, MPI_Info info, OSHMPI_ictx_t * ictx)
{
    OSHMPI_CALLMPI(MPI_Win_create(base, size, 1 /* disp_unit */ , info,
                                  OSHMPI_global.comm_world, &ictx->win));
    OSHMPI_CALLMPI(MPI_Win_lock_all(MPI_MODE_NOCHECK, ictx->win));
    OSHMPIU_ATOMIC_FLAG_STORE(ictx->outstanding_op, 0);
    OSHMPI_ICTX_SET_DISP_MODE(ictx, OSHMPI_RELATIVE_DISP);
}

static void space_ictx_destroy(OSHMPI_ictx_t * ictx)
{
    OSHMPI_ASSERT(ictx->win != MPI_WIN_NULL);
    OSHMPI_CALLMPI(MPI_Win_unlock_all(ictx->win));
    OSHMPI_CALLMPI(MPI_Win_free(&ictx->win));
    ictx->win = MPI_WIN_NULL;
}

#ifndef OSHMPI_DISABLE_DEBUG
static const char *space_memkind_str(shmemx_memkind_t memkind)
{
    switch (memkind) {
        case SHMEMX_MEM_CUDA:
            return "cuda";
            break;
        case SHMEMX_MEM_ZE:
            return "ze";
            break;
        case SHMEMX_MEM_HOST:
        default:
            return "host";
            break;
    }
}
#endif

#ifdef OSHMPI_ENABLE_CUDA
static void *space_cuda_malloc(MPI_Aint size)
{
    void *base = NULL;
    OSHMPI_CALLCUDA(cudaMalloc(&base, size));
    return base;
}

static void space_cuda_free(void *base)
{
    OSHMPI_CALLCUDA(cudaFree(base));
}
#else
static void *space_cuda_malloc(MPI_Aint size)
{
    OSHMPI_ERR_ABORT("Memory kind CUDA is disabled. Recompile with --enable-cuda to enable\n");
    return NULL;
}

static void space_cuda_free(void *base)
{
    OSHMPI_ERR_ABORT("Memory kind CUDA is disabled. Recompile with --enable-cuda to enable\n");
}
#endif

#ifdef OSHMPI_ENABLE_ZE
static void *space_ze_malloc(MPI_Aint size, shmemx_device_handle_t device_handle)
{
    ze_device_mem_alloc_desc_t device_desc = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        .pNext = NULL,
        .flags = 0,
        .ordinal = 0,   /* We currently support a single memory type */
    };
    /* Currently ZE ignores this argument and uses an internal alignment
     * value. However, this behavior can change in the future. */
    size_t mem_alignment = 1;
    void *ptr;

    ze_result_t ret = zeMemAllocDevice(global_ze_context, &device_desc, size, mem_alignment, device_handle, &ptr);
    OSHMPI_ASSERT(ret == ZE_RESULT_SUCCESS);

    return ptr;
}

static void space_ze_free(void *base)
{
    ze_result_t ret = zeMemFree(global_ze_context, base);
    OSHMPI_ASSERT(ret == ZE_RESULT_SUCCESS);
}
#else
static void *space_ze_malloc(MPI_Aint size, shmemx_device_handle_t device_handle)
{
    OSHMPI_ERR_ABORT("Memory kind ZE is disabled. Recompile with --enable-ze to enable\n");
    return NULL;
}

static void space_ze_free(void *base)
{
    OSHMPI_ERR_ABORT("Memory kind ZE is disabled. Recompile with --enable-ze to enable\n");
}
#endif

void OSHMPI_space_initialize(void)
{
    OSHMPIU_gpu_init();
    OSHMPI_THREAD_INIT_CS(&OSHMPI_global.space_list.cs);
}

void OSHMPI_space_finalize(void)
{
    OSHMPI_THREAD_DESTROY_CS(&OSHMPI_global.space_list.cs);
    OSHMPIU_gpu_finalize();
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
    MPI_Aint size = OSHMPI_ALIGN(space_config.sheap_size, OSHMPI_global.page_sz);
    void *base = NULL;

    switch (space_config.memkind) {
        case SHMEMX_MEM_CUDA:
            base = space_cuda_malloc(size);
            break;
        case SHMEMX_MEM_ZE:
            base = space_ze_malloc(size, space_config.device_handle);
            break;
        case SHMEMX_MEM_HOST:
        default:
            base = OSHMPIU_malloc(size);
            space_config.memkind = SHMEMX_MEM_HOST;
            break;
    }
    OSHMPI_ASSERT(base);

    /* Initialize memory pool per space */
    OSHMPIU_mempool_init(&space->mem_pool, base, size, OSHMPI_global.page_sz);
    OSHMPI_THREAD_INIT_CS(&space->mem_pool_cs);

    OSHMPI_sobj_init_attr(&space->sobj_attr, space_config.memkind, base, size);
    OSHMPI_sobj_set_handle(&space->sobj_attr, OSHMPI_SOBJ_SPACE_HEAP, 0, 0);

    OSHMPI_THREAD_ENTER_CS(&OSHMPI_global.space_list.cs);
    LL_APPEND(OSHMPI_global.space_list.head, space);
    OSHMPI_global.space_list.nspaces++;
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_global.space_list.cs);

    space->ctx_list = NULL;
    space->config = space_config;
#ifndef OSHMPI_ENABLE_DYNAMIC_WIN
    space->default_ictx.win = MPI_WIN_NULL;
    OSHMPIU_ATOMIC_FLAG_STORE(space->default_ictx.outstanding_op, 0);
#endif

    OSHMPI_DBGMSG
        ("create space %p, base %p, size %ld, num_contexts=%d, memkind=%d (%s), handle 0x%x\n",
         space, space->sobj_attr.base, space->sobj_attr.size, space->config.num_contexts,
         space->config.memkind, space_memkind_str(space->config.memkind), space->sobj_attr.handle);

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
            space_cuda_free(space->sobj_attr.base);
            break;
        case SHMEMX_MEM_ZE:
            space_ze_free(space->sobj_attr.base);
            break;
        case SHMEMX_MEM_HOST:
        default:
            OSHMPIU_free(space->sobj_attr.base);
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
        if (OSHMPIU_ATOMIC_FLAG_CAS(space->ctx_list[i].used_flag, 0, 1) == 0) {
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
    int symm_flag = 0;
    OSHMPI_sobj_symm_info_allgather(&space->sobj_attr, &symm_flag);
    OSHMPI_sobj_set_handle(&space->sobj_attr, OSHMPI_SOBJ_SPACE_ATTACHED_HEAP,
                           symm_flag, space_attach_idx);
    space_attach_idx++;

    /* Create internal window */
#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
    OSHMPI_CALLMPI(MPI_Win_attach(OSHMPI_global.symm_ictx.win, space->sobj_attr.base,
                                  (MPI_Aint) space->sobj_attr.size));
    OSHMPI_am_progress_mpi_barrier(OSHMPI_global.comm_world);
    OSHMPI_DBGMSG("attach space %p, default ctx: attach to symm_ictx\n", space);
#else
    space_ictx_create(space->sobj_attr.base, (MPI_Aint) space->sobj_attr.size, info,
                      &space->default_ictx);
    OSHMPI_DBGMSG("attach space %p, default ctx: new ictx.win 0x%lx\n", space,
                  (uint64_t) space->default_ictx.win);
#endif
    OSHMPI_sobj_symm_info_dbgprint(&space->sobj_attr);

    /* TODO: assume all processes have the same config */
    /* Create explicit-context windows */
    if (space->config.num_contexts > 0) {
        space->ctx_list =
            (OSHMPI_ctx_t *) OSHMPIU_malloc(sizeof(OSHMPI_ctx_t) * space->config.num_contexts);
        int i;
        for (i = 0; i < space->config.num_contexts; i++) {
            space_ictx_create(space->sobj_attr.base, (MPI_Aint) space->sobj_attr.size,
                              info, &space->ctx_list[i].ictx);

            /* copy into context to avoid pointer dereference in RMA/AMO path */
            space->ctx_list[i].sobj_attr = space->sobj_attr;
            OSHMPIU_ATOMIC_FLAG_STORE(space->ctx_list[i].used_flag, 0);

            OSHMPI_DBGMSG("attach space %p, private ctx[%d]: new ictx.win 0x%lx\n",
                          space, i, (uint64_t) space->ctx_list[i].ictx.win);
        }
    }

    OSHMPI_CALLMPI(MPI_Info_free(&info));
}

/* Collectively detach the space from the default team */
void OSHMPI_space_detach(OSHMPI_space_t * space)
{
    int i;

#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
    OSHMPI_am_progress_mpi_barrier(OSHMPI_global.comm_world);
    OSHMPI_CALLMPI(MPI_Win_detach(OSHMPI_global.symm_ictx.win, space->sobj_attr.base));
#else
    /* Destroy internal window */
    space_ictx_destroy(&space->default_ictx);
#endif

    /* Space should have already be attached or no context is required at config */
    OSHMPI_ASSERT((space->config.num_contexts == 0 || space->ctx_list));

    /* Destroy explicit-context windows */
    for (i = 0; i < space->config.num_contexts; i++) {
        OSHMPI_ASSERT(OSHMPIU_ATOMIC_FLAG_LOAD(space->ctx_list[i].used_flag) == 0);
        space_ictx_destroy(&space->ctx_list[i].ictx);
    }
    OSHMPIU_free(space->ctx_list);
    space->ctx_list = NULL;

    OSHMPI_sobj_destroy_attr(&space->sobj_attr);
    OSHMPI_sobj_set_handle(&space->sobj_attr, OSHMPI_SOBJ_SPACE_HEAP, 0, 0);
}

/* Collectively allocate a buffer from the space */
void *OSHMPI_space_malloc(OSHMPI_space_t * space, size_t size)
{
    void *ptr = NULL;

    OSHMPI_THREAD_ENTER_CS(&space->mem_pool_cs);
    ptr = OSHMPIU_mempool_alloc(&space->mem_pool, size);
    OSHMPI_THREAD_EXIT_CS(&space->mem_pool_cs);

    OSHMPI_DBGMSG("space_malloc from space %p, size %ld -> ptr %p, disp 0x%lx\n",
                  space, size, ptr, (MPI_Aint) ptr - (MPI_Aint) space->sobj_attr.base);
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
                  space, ptr, (MPI_Aint) ptr - (MPI_Aint) space->sobj_attr.base);
}

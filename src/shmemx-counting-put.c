#include "shmemx.h"
#include "shmem-internals.h"

void shmemx_ct_create(shmemx_ct_t *ct)
{
    *ct = shmalloc(sizeof(long));
}

void shmemx_ct_free(shmemx_ct_t *ct)
{
    shfree(*ct);
}

long shmemx_ct_get(shmemx_ct_t ct)
{
#ifdef ENABLE_SMP_OPTIMIZATIONS
    if (shmem_world_is_smp) {
        return __sync_fetch_and_add(ct,0);
    } else  
#endif
    {
        shmem_offset_t win_offset = (ptrdiff_t)((intptr_t)ct - (intptr_t)shmem_sheap_base_ptr);
        long output;
        MPI_Fetch_and_op(NULL, &output, MPI_LONG, shmem_world_rank, win_offset, MPI_NO_OP, shmem_sheap_win);
        MPI_Win_flush_local(shmem_world_rank, shmem_sheap_win); 
        return output;
    }   
}

void shmemx_ct_set(shmemx_ct_t ct, long value)
{
#ifdef ENABLE_SMP_OPTIMIZATIONS
    if (shmem_world_is_smp) {
        __sync_lock_test_and_set(ct,value);
    } else  
#endif
    {
        shmem_offset_t win_offset = (ptrdiff_t)((intptr_t)ct - (intptr_t)shmem_sheap_base_ptr);
        MPI_Fetch_and_op(&value, NULL, MPI_LONG, shmem_world_rank, win_offset, MPI_REPLACE, shmem_sheap_win);
        MPI_Win_flush(shmem_world_rank, shmem_sheap_win); 
    }
    return;
}

void shmemx_ct_wait(shmemx_ct_t ct, long wait_for)
{
    while (wait_for != shmemx_ct_get(ct));
}

void shmemx_putmem_ct(shmemx_ct_t ct, void *target, const void *source, size_t len, int pe)
{
    shmem_putmem(target, source, len, pe);
    __shmem_remote_sync_pe(pe);
    shmem_long_add(ct, 1, pe);
}


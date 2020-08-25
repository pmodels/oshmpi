/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include "oshmpi_util.h"
#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif /* HAVE_SYS_MMAN_H */
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif /* HAVE_SYS_TIME_H */
#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif /* HAVE_SYS_STAT_H */
#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif /* HAVE_FCNTL_H */
#include <stdint.h>
#include <unistd.h>
#include <errno.h>

#define __ALLOC_SYMM_MEM_RETRY 100
#define __RANDOM_ADDR_RETRY 100

static struct {
    size_t page_sz;
    MPI_Comm comm_world;
    int world_rank;
    int world_size;
} symm_mem_global;

/* Check if the specified memory range is available. Return 1 if
 * yes, otherwise 0.*/
static inline int check_maprange_ok(void *start, size_t mapsize)
{
    int rc = 0;
    int ret = 1;
    size_t i, num_pages = mapsize / symm_mem_global.page_sz;
    char *ptr = (char *) start;

    for (i = 0; i < num_pages; i++) {
        rc = msync(ptr, symm_mem_global.page_sz, 0);

        if (rc == -1) {
            OSHMPI_ASSERT(errno == ENOMEM);
            ptr += symm_mem_global.page_sz;
        } else {
            ret = 0;    /* the range has already been used */
            break;
        }
    }

    return ret;
}

static inline void *generate_random_addr(size_t mapsize)
{
    /* starting position for pointer to map
     * This is not generic, probably only works properly on Linux
     * but it's not fatal since we bail after a fixed number of iterations */
#define __MAP_POINTER(random_unsigned)    \
        ((random_unsigned&((0x00006FFFFFFFFFFF&(~(symm_mem_global.page_sz-1)))|0x0000600000000000)))
    uintptr_t map_pointer;

    char random_state[256];
    uint64_t random_unsigned;
    struct timeval ts;
    int iter = __RANDOM_ADDR_RETRY;
    int32_t rh, rl;
    struct random_data rbuf;

    /* rbuf must be zero-cleared otherwise it results in SIGSEGV in glibc
     * (http://stackoverflow.com/questions/4167034/c-initstate-r-crashing) */
    memset(&rbuf, 0, sizeof(rbuf));
    gettimeofday(&ts, NULL);

    initstate_r(ts.tv_usec, random_state, sizeof(random_state), &rbuf);
    random_r(&rbuf, &rh);
    random_r(&rbuf, &rl);
    random_unsigned = ((uint64_t) rh) << 32 | (uint64_t) rl;
    map_pointer = __MAP_POINTER(random_unsigned);

    while (!check_maprange_ok((void *) map_pointer, mapsize)) {
        random_r(&rbuf, &rh);
        random_r(&rbuf, &rl);
        random_unsigned = ((uint64_t) rh) << 32 | (uint64_t) rl;
        map_pointer = __MAP_POINTER(random_unsigned);
        iter--;
        if (iter == 0) {
            map_pointer = -1ULL;
            break;
        }
    }

    return (void *) map_pointer;
}

/* Collectively allocate a symmetric memory region over all processes.
 * It internally tries multiple times and return 0 if succeed, otherwise
 * return 1.
 *
 * Parameters:
 * IN size: size of the memory region on each process. The caller ensures that
 *          the size is page aligned.
 * OUT local_addr_ptr: start address of the allocated region. NULL if allocation fails. */
int OSHMPIU_allocate_symm_mem(MPI_Aint size, void **local_addr_ptr)
{
#ifdef OSHMPI_ENABLE_SYMM_ALLOC
    int iter = __ALLOC_SYMM_MEM_RETRY;
    unsigned any_mapfail_flag = 1;
    void *local_addr = NULL;

    while (any_mapfail_flag && iter-- > 0) {
        void *map_pointer = NULL;
        unsigned mapfail_flag = 0;

        /* the leading process in win get a random address */
        if (symm_mem_global.world_rank == 0)
            map_pointer = generate_random_addr(size);

        /* broadcast fixed address to the other processes in win */
        OSHMPI_CALLMPI(MPI_Bcast(&map_pointer, sizeof(map_pointer), MPI_CHAR, 0,
                                 symm_mem_global.comm_world));

        /* return failure if the leading process cannot generate a valid address */
        if (map_pointer == (void *) -1ULL)
            break;

        int rc = check_maprange_ok(map_pointer, size);
        if (rc) {
            local_addr = mmap(map_pointer, size,
                              PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
        } else
            local_addr = (void *) MAP_FAILED;
        mapfail_flag = (local_addr == (void *) MAP_FAILED) ? 1 : 0;

        /* check if any mapping failure occurs */
        OSHMPI_CALLMPI(MPI_Allreduce(&mapfail_flag, &any_mapfail_flag, 1, MPI_UNSIGNED,
                                     MPI_BOR, symm_mem_global.comm_world));

        /* cleanup local shm segment if mapping failed */
        if (any_mapfail_flag)
            munmap(local_addr, size);
    }

    if (!any_mapfail_flag)
        *local_addr_ptr = local_addr;
    return any_mapfail_flag;
#else
    *local_addr_ptr = NULL;
    return 1;
#endif
}

void OSHMPIU_free_symm_mem(void *local_addr, MPI_Aint size)
{
    munmap(local_addr, size);
}

/* Collects the starting address of the memory region from all processes
 * and check if all addresses are the same (symmetric). We skip the check
 * on size because OSHMPI always allocates memory regions with the same
 * size (e.g., symm heap).
 *
 * Parameters:
 * IN  local_addr: starting address of the local memory region
 * OUT symm_flag_ptr: 1 if symmetric otherwise 0
 * OUT all_addrs_ptr: the routine internally allocates the buffer for storing
 *                    all addresses and return to the caller if the addresses
 *                    are not symmetric. The caller needs to free it after use.
 *                    If the addresses are symmetric, then the buffer is internally freed. */
void OSHMPIU_check_symm_mem(void *local_addr, int *symm_flag_ptr, MPI_Aint ** all_addrs_ptr)
{
    int i;
    int symm_flag = 1;
    MPI_Aint *all_addrs = NULL;

    all_addrs = OSHMPIU_malloc(sizeof(MPI_Aint) * symm_mem_global.world_size);
    OSHMPI_ASSERT(all_addrs != NULL);

    OSHMPI_CALLMPI(MPI_Get_address((const void *) local_addr,
                                   &all_addrs[symm_mem_global.world_rank]));
    OSHMPI_CALLMPI(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                 all_addrs, sizeof(MPI_Aint), MPI_BYTE,
                                 symm_mem_global.comm_world));

    for (i = 0; i < symm_mem_global.world_size; i++) {
        if (all_addrs[i] != all_addrs[symm_mem_global.world_rank]) {
            symm_flag = 0;
            break;
        }
    }

    /* Store all addresses if it is non-symmetric */
    if (symm_flag)
        OSHMPIU_free(all_addrs);
    else
        *all_addrs_ptr = all_addrs;

    *symm_flag_ptr = symm_flag;
}

void OSHMPIU_initialize_symm_mem(MPI_Comm comm_world)
{
    symm_mem_global.comm_world = comm_world;
    OSHMPI_CALLMPI(MPI_Comm_rank(comm_world, &symm_mem_global.world_rank));
    OSHMPI_CALLMPI(MPI_Comm_size(comm_world, &symm_mem_global.world_size));
    symm_mem_global.page_sz = (size_t) sysconf(_SC_PAGESIZE);
}

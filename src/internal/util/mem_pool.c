/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include "oshmpi_util.h"

/* Initialize memory pool structure by using user provided
 * mem_pool object, memory region starting address, memory region size,
 * and chunk size. The memory region size is required to align with the
 * chunk size.*/
void OSHMPIU_mempool_init(OSUMPIU_mempool_t * mem_pool, void *base, size_t aligned_mem_size,
                          size_t chunk_size)
{
    mem_pool->base = base;
    mem_pool->size = aligned_mem_size;
    mem_pool->chunk_size = chunk_size;
    mem_pool->nchunks = aligned_mem_size / chunk_size;
    mem_pool->chunks_nused =
        (unsigned int *) OSHMPIU_malloc(sizeof(unsigned int) * mem_pool->nchunks);
    OSHMPI_ASSERT(mem_pool->chunks_nused);

    memset(mem_pool->chunks_nused, 0, sizeof(unsigned int) * mem_pool->nchunks);
}

/* Destroy the memory pool internal structure.
 * The memory region and mem_pool itself are freed by user. */
void OSHMPIU_mempool_destroy(OSUMPIU_mempool_t * mem_pool)
{
    int i;
    for (i = 0; i < mem_pool->nchunks; i++)
        OSHMPI_ASSERT(mem_pool->chunks_nused[i] == 0);

    OSHMPIU_free(mem_pool->chunks_nused);
}

/* Find a contiguous free region from the memory pool. If found, the
 * pointer of the starting address is returned; otherwise NULL is returned. */
void *OSHMPIU_mempool_alloc(OSUMPIU_mempool_t * mem_pool, size_t size)
{
    int off = 0, navail = 0;
    void *ptr = NULL;

    size_t aligned_sz = OSHMPI_ALIGN(size, mem_pool->chunk_size);
    int nalloc = aligned_sz / mem_pool->chunk_size;

    while (off < mem_pool->nchunks) {
        unsigned int nused = mem_pool->chunks_nused[off];
        /* Skip nused number of chunks as they are already used.
         * Also reset navail because we need allocate contiguous region */
        if (nused) {
            navail = 0;
            off += nused;
            continue;
        }

        navail++;       /* Count available chunk */
        off++;

        if (navail == nalloc)   /* Found enough contiguous chunks */
            break;
    }

    if (navail == nalloc) {
        int first_off = off - navail;
        ptr = (char *) mem_pool->base + first_off * mem_pool->chunk_size;
        mem_pool->chunks_nused[first_off] = nalloc;
    }

    return ptr;
}

/* Free an allocated memory region in the memory pool. */
void OSHMPIU_mempool_free(OSUMPIU_mempool_t * mem_pool, void *ptr)
{
    /* Calculate the offset of the first chunk associated with ptr.
     * Ptr should be always match the start address of a chunk. */
    int off = ((char *) ptr - (char *) mem_pool->base) / mem_pool->chunk_size;
    OSHMPI_ASSERT(off >= 0 && off < mem_pool->nchunks && mem_pool->chunks_nused[off]);
    mem_pool->chunks_nused[off] = 0;
}

/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_UTIL_MEMPOOL_H
#define INTERNAL_UTIL_MEMPOOL_H

/* Memory pool utility routines (thread-unsafe). */
#define MEMPOOL_DYNAMIC_NOBJS_PER_BLOCK 64

typedef enum {
    OSHMPI_MEMPOOL_PERALLOC_OBJ,
    OSHMPI_MEMPOOL_DYNAMIC_OBJ,
} OSHMPI_mempool_objkind_t;

/* Header variables that should be include and defined as the first member
 * of any memory pool object type */
#define OSHMPI_MEMPOOL_OBJ_HEADER \
    OSHMPI_mempool_objkind_t kind;

typedef struct OSHMPI_mempool_obj {
    OSHMPI_MEMPOOL_OBJ_HEADER struct OSHMPI_mempool_obj *next;
} OSHMPI_mempool_obj_t;

typedef struct OSHMPI_mempool_block {
    void *objs_addr;
    struct OSHMPI_mempool_block *next;
} OSHMPI_mempool_block_t;

typedef struct OSHMPI_mempool {
    OSHMPI_mempool_obj_t *avail_head;   /* List of available objects */
    int initialized_flag;
    size_t size;                /* Size of an individual object */
    OSHMPI_mempool_block_t *dynamic_head;       /* List of dynamic object blocks */
    int dyanmic_nblks;          /* Number of allocated dynamic objects */
    void *prealloc;             /* Pointer to preallocated block */
    int prealloc_nobjs;
} OSHMPI_mempool_t;

OSHMPI_STATIC_INLINE_PREFIX void mempool_prepend_avail_objs(OSHMPI_mempool_t * pool,
                                                            char *objs_addr, int nobjs,
                                                            OSHMPI_mempool_objkind_t kind)
{
    int i;
    OSHMPI_mempool_obj_t *obj = NULL;

    for (i = 0; i < nobjs; i++) {
        obj = (OSHMPI_mempool_obj_t *) (void *) objs_addr;
        obj->kind = kind;
        LL_PREPEND(pool->avail_head, obj);
        objs_addr += pool->size;
    }
}

OSHMPI_STATIC_INLINE_PREFIX void mempool_initialize_prealloc(OSHMPI_mempool_t * pool,
                                                             OSHMPI_mempool_obj_t ** obj_ptr)
{
    char *ptr = (char *) pool->prealloc;

    OSHMPI_ASSERT(pool->prealloc_nobjs > 0);

    /* Return first object */
    *obj_ptr = (OSHMPI_mempool_obj_t *) (void *) ptr;
    ptr += pool->size;

    /* Enqueue remaining preallocated objects */
    mempool_prepend_avail_objs(pool, ptr, pool->prealloc_nobjs - 1, OSHMPI_MEMPOOL_PERALLOC_OBJ);

    pool->initialized_flag = 1;
}

OSHMPI_STATIC_INLINE_PREFIX void mempool_realloc_dynamic(OSHMPI_mempool_t * pool,
                                                         OSHMPI_mempool_obj_t ** obj_ptr)
{
    OSHMPI_mempool_block_t *block = NULL;

    block =
        OSHMPIU_malloc(sizeof(OSHMPI_mempool_block_t) +
                       pool->size * MEMPOOL_DYNAMIC_NOBJS_PER_BLOCK);
    OSHMPI_ASSERT(block);

    block->objs_addr = (char *) block + sizeof(OSHMPI_mempool_block_t);

    LL_PREPEND(pool->dynamic_head, block);
    pool->dyanmic_nblks++;

    char *ptr = (char *) block->objs_addr;

    /* Reallocate dynamic objects only when available objects are used up.
     * Thus return first object */
    *obj_ptr = (OSHMPI_mempool_obj_t *) (void *) ptr;
    ptr += pool->size;

    /* Enqueue remaining objects */
    mempool_prepend_avail_objs(pool, ptr, MEMPOOL_DYNAMIC_NOBJS_PER_BLOCK - 1,
                               OSHMPI_MEMPOOL_DYNAMIC_OBJ);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPIU_mempool_destroy(OSHMPI_mempool_t * pool)
{
    OSHMPI_mempool_block_t *block = NULL, *tmp = NULL;

    LL_FOREACH_SAFE(pool->dynamic_head, block, tmp) {
        LL_DELETE(pool->dynamic_head, block);
        OSHMPIU_free(block);
        pool->dyanmic_nblks--;
    }
    OSHMPI_ASSERT(pool->dyanmic_nblks == 0);
}


OSHMPI_STATIC_INLINE_PREFIX void OSHMPIU_mempool_initialize(OSHMPI_mempool_t * pool, size_t size,
                                                            void *prealloc_addr, int prealloc_nobjs)
{
    pool->avail_head = NULL;
    pool->dyanmic_nblks = 0;
    pool->dynamic_head = NULL;
    pool->initialized_flag = 0;
    pool->prealloc = prealloc_addr;
    pool->prealloc_nobjs = prealloc_nobjs;
    pool->size = size;
}

OSHMPI_STATIC_INLINE_PREFIX void *OSHMPIU_mempool_alloc_obj(OSHMPI_mempool_t * pool)
{
    OSHMPI_mempool_obj_t *obj = NULL;

    if (pool->avail_head) {
        obj = pool->avail_head;
        pool->avail_head = obj->next;
    } else {
        /* Pool is not initialized. Setup preallocated objects */
        if (!pool->initialized_flag) {
            mempool_initialize_prealloc(pool, &obj);
        } else {
            /* Existing objects are used up. Reallocate a new dynamic chunk */
            mempool_realloc_dynamic(pool, &obj);
        }
    }

    return (void *) obj;
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPIU_mempool_free_obj(OSHMPI_mempool_t * pool, void *obj)
{
    LL_PREPEND(pool->avail_head, (OSHMPI_mempool_obj_t *) obj);
}
#endif /* INTERNAL_UTIL_MEMPOOL_H */

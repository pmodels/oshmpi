/* The following lines are automatically generated. DO NOT EDIT. */
/* TPL_BLOCK_START */
TYPE shmem_TYPENAME_atomic_compare_swap(TYPE * dest, TYPE cond, TYPE value, int pe);
TYPE shmem_ctx_TYPENAME_atomic_compare_swap(shmem_ctx_t ctx, TYPE * dest, TYPE cond, TYPE
                                            value, int pe);
TYPE shmem_TYPENAME_atomic_fetch_inc(TYPE * dest, int pe);
TYPE shmem_ctx_TYPENAME_atomic_fetch_inc(shmem_ctx_t ctx, TYPE * dest, int pe);
void shmem_TYPENAME_atomic_inc(TYPE * dest, int pe);
void shmem_ctx_TYPENAME_atomic_inc(shmem_ctx_t ctx, TYPE * dest, int pe);
TYPE shmem_TYPENAME_atomic_fetch_add(TYPE * dest, TYPE value, int pe);
TYPE shmem_ctx_TYPENAME_atomic_fetch_add(shmem_ctx_t ctx, TYPE * dest, TYPE value, int pe);
void shmem_TYPENAME_atomic_add(TYPE * dest, TYPE value, int pe);
void shmem_ctx_TYPENAME_atomic_add(shmem_ctx_t ctx, TYPE * dest, TYPE value, int pe);
/* Deprecated APIs start */
TYPE shmem_TYPENAME_cswap(TYPE * dest, TYPE cond, TYPE value, int pe);
TYPE shmem_TYPENAME_finc(TYPE * dest, int pe);
void shmem_TYPENAME_inc(TYPE * dest, int pe);
TYPE shmem_TYPENAME_fadd(TYPE * dest, TYPE value, int pe);
void shmem_TYPENAME_add(TYPE * dest, TYPE value, int pe);
/* Deprecated APIs end */
/* TPL_BLOCK_END */
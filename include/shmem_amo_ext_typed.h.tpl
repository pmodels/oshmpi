/* TPL_HEADER_START */
/* The following lines are automatically generated. DO NOT EDIT. */
/* TPL_HEADER_END */
TYPE shmem_TYPENAME_atomic_fetch(const TYPE * source, int pe);
TYPE shmem_ctx_TYPENAME_atomic_fetch(shmem_ctx_t ctx, const TYPE * source, int pe);
void shmem_TYPENAME_atomic_set(TYPE * dest, TYPE value, int pe);
void shmem_ctx_TYPENAME_atomic_set(shmem_ctx_t ctx, TYPE * dest, TYPE value, int pe);
TYPE shmem_TYPENAME_atomic_swap(TYPE * dest, TYPE value, int pe);
TYPE shmem_ctx_TYPENAME_atomic_swap(shmem_ctx_t ctx, TYPE * dest, TYPE value, int pe);
/* Deprecated APIs start */
TYPE shmem_TYPENAME_fetch(const TYPE * source, int pe);
void shmem_TYPENAME_set(TYPE * dest, TYPE value, int pe);
TYPE shmem_TYPENAME_swap(TYPE * dest, TYPE value, int pe);
/* Deprecated APIs end */

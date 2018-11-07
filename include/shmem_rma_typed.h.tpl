/* The following lines are automatically generated. DO NOT EDIT. */
/* TPL_BLOCK_START */
TYPE shmem_TYPENAME_g(const TYPE * source, int pe);
TYPE shmem_ctx_TYPENAME_g(shmem_ctx_t ctx, const TYPE * source, int pe);
void shmem_TYPENAME_get(TYPE * dest, const TYPE * source, size_t nelems, int pe);
void shmem_ctx_TYPENAME_get(shmem_ctx_t ctx, TYPE * dest, const TYPE * source,
                            size_t nelems, int pe);
void shmem_TYPENAME_get_nbi(TYPE * dest, const TYPE * source, size_t nelems, int pe);
void shmem_ctx_TYPENAME_get_nbi(shmem_ctx_t ctx, TYPE * dest, const TYPE * source,
                                size_t nelems, int pe);
void shmem_TYPENAME_iget(TYPE * dest, const TYPE * source, ptrdiff_t dst, ptrdiff_t sst,
                         size_t nelems, int pe);
void shmem_ctx_TYPENAME_iget(shmem_ctx_t ctx, TYPE * dest, const TYPE * source, ptrdiff_t dst,
                             ptrdiff_t sst, size_t nelems, int pe);
void shmem_TYPENAME_p(TYPE * dest, TYPE value, int pe);
void shmem_ctx_TYPENAME_p(shmem_ctx_t ctx, TYPE * dest, TYPE value, int pe);
void shmem_TYPENAME_put(TYPE * dest, const TYPE * source, size_t nelems, int pe);
void shmem_ctx_TYPENAME_put(shmem_ctx_t ctx, TYPE * dest, const TYPE * source,
                            size_t nelems, int pe);
void shmem_TYPENAME_iput(TYPE * dest, const TYPE * source, ptrdiff_t dst,
                         ptrdiff_t sst, size_t nelems, int pe);
void shmem_ctx_TYPENAME_iput(shmem_ctx_t ctx, TYPE * dest, const TYPE * source,
                             ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_TYPENAME_put_nbi(TYPE * dest, const TYPE * source, size_t nelems, int pe);
void shmem_ctx_TYPENAME_put_nbi(shmem_ctx_t ctx, TYPE * dest, const TYPE * source,
                                size_t nelems, int pe);
/* TPL_BLOCK_END */
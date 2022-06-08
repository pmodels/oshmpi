/* The following lines are automatically generated. DO NOT EDIT. */
/* TPL_BLOCK_START */

void shmem_TYPENAME_put_signal(TYPE * dest, const TYPE * source, size_t nelems, uint64_t *sig_addr,
                               uint64_t signal, int sig_op, int pe);
void shmem_ctx_TYPENAME_put_signal(shmem_ctx_t ctx, TYPE * dest, const TYPE * source, size_t nelems,
                                   uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
void shmem_TYPENAME_put_signal_nbi(TYPE * dest, const TYPE * source, size_t nelems,
                                   uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
void shmem_ctx_TYPENAME_put_signal_nbi(shmem_ctx_t ctx, TYPE * dest, const TYPE * source,
                                       size_t nelems, uint64_t *sig_addr, uint64_t signal,
                                       int sig_op, int pe);
/* TPL_BLOCK_END */

/* *INDENT-OFF* */
#if OSHMPI_HAVE_C11
#define shmem_put_signal(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_put_signal,   \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_put_signal,       \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore     \
    )(__VA_ARGS__)

#define shmem_put_signal_nbi(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_put_signal_nbi, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore     \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_put_signal_nbi,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore         \
    )(__VA_ARGS__)
#endif /* OSHMPI_HAVE_C11 */
/* *INDENT-ON* */

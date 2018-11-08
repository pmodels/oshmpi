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

/* *INDENT-OFF* */
#if OSHMPI_HAVE_C11
#define shmem_g(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_g,   \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_g,       \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore     \
    )(__VA_ARGS__)

#define shmem_get(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_get, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_get,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore     \
    )(__VA_ARGS__)

#define shmem_get_nbi(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_get_nbi, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore     \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_get_nbi,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore         \
    )(__VA_ARGS__)

#define shmem_iget(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_iget, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore  \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_iget,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore      \
    )(__VA_ARGS__)

#define shmem_p(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_p,   \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_p,       \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore     \
    )(__VA_ARGS__)

#define shmem_put(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_put, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_put,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore     \
    )(__VA_ARGS__)

#define shmem_put_nbi(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_put_nbi, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore     \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_put_nbi,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore         \
    )(__VA_ARGS__)

#define shmem_iput(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_iput, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore  \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_iput,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore      \
    )(__VA_ARGS__)
#endif /* OSHMPI_HAVE_C11 */
/* *INDENT-ON* */

/* The following lines are automatically generated. DO NOT EDIT. */
/* TPL_BLOCK_START */
TYPE shmem_TYPENAME_atomic_fetch(const TYPE * source, int pe);
TYPE shmem_ctx_TYPENAME_atomic_fetch(shmem_ctx_t ctx, const TYPE * source, int pe);
void shmem_TYPENAME_atomic_set(TYPE * dest, TYPE value, int pe);
void shmem_ctx_TYPENAME_atomic_set(shmem_ctx_t ctx, TYPE * dest, TYPE value, int pe);
TYPE shmem_TYPENAME_atomic_swap(TYPE * dest, TYPE value, int pe);
TYPE shmem_ctx_TYPENAME_atomic_swap(shmem_ctx_t ctx, TYPE * dest, TYPE value, int pe);

TYPE shmem_TYPENAME_atomic_fetch_nbi(TYPE * fetch, const TYPE * source, int pe);
TYPE shmem_ctx_TYPENAME_atomic_fetch_nbi(shmem_ctx_t ctx, TYPE * fetch, const TYPE * source,
                                         int pe);
TYPE shmem_TYPENAME_atomic_swap_nbi(TYPE * fetch, TYPE * dest, TYPE value, int pe);
TYPE shmem_ctx_TYPENAME_atomic_swap_nbi(shmem_ctx_t ctx, TYPE * fetch, TYPE * dest, TYPE value,
                                        int pe);
/* Deprecated APIs start */
TYPE shmem_TYPENAME_fetch(const TYPE * source, int pe);
void shmem_TYPENAME_set(TYPE * dest, TYPE value, int pe);
TYPE shmem_TYPENAME_swap(TYPE * dest, TYPE value, int pe);
/* Deprecated APIs end */
/* TPL_BLOCK_END */

/* *INDENT-OFF* */
#if OSHMPI_HAVE_C11
#define shmem_atomic_fetch(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_fetch, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore          \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_fetch,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_fetch_nbi(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_fetch_nbi, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore          \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_fetch_nbi,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_set(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_set, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore        \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_set,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_swap(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_swap, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore         \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_swap,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_swap_nbi(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_swap_nbi, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore         \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_swap_nbi,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

/* Deprecated APIs start */
#define shmem_fetch(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_fetch, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore   \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_fetch,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_set(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_set, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_set,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_swap(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_swap, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore  \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_swap,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)
/* Deprecated APIs end */

#endif /* OSHMPI_HAVE_C11 */
/* *INDENT-ON* */

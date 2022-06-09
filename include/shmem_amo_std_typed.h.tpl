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

TYPE shmem_TYPENAME_atomic_compare_swap_nbi(TYPE * fetch,TYPE * dest, TYPE cond, TYPE value,
                                            int pe);
TYPE shmem_ctx_TYPENAME_atomic_compare_swap_nbi(shmem_ctx_t ctx, TYPE * fetch, TYPE * dest,
                                                TYPE cond, TYPE value, int pe);
TYPE shmem_TYPENAME_atomic_fetch_inc_nbi(TYPE * fetch, TYPE * dest, int pe);
TYPE shmem_ctx_TYPENAME_atomic_fetch_inc_nbi(shmem_ctx_t ctx, TYPE * fetch, TYPE * dest, int pe);
TYPE shmem_TYPENAME_atomic_fetch_add_nbi(TYPE * fetch, TYPE * dest, TYPE value, int pe);
TYPE shmem_ctx_TYPENAME_atomic_fetch_add_nbi(shmem_ctx_t ctx, TYPE * fetch, TYPE * dest, TYPE value,
                                             int pe);
/* Deprecated APIs start */
TYPE shmem_TYPENAME_cswap(TYPE * dest, TYPE cond, TYPE value, int pe);
TYPE shmem_TYPENAME_finc(TYPE * dest, int pe);
void shmem_TYPENAME_inc(TYPE * dest, int pe);
TYPE shmem_TYPENAME_fadd(TYPE * dest, TYPE value, int pe);
void shmem_TYPENAME_add(TYPE * dest, TYPE value, int pe);
/* Deprecated APIs end */
/* TPL_BLOCK_END */

/* *INDENT-OFF* */
#if OSHMPI_HAVE_C11
#define shmem_atomic_compare_swap(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_compare_swap, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore                 \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_compare_swap,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_compare_swap_nbi(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_compare_swap_nbi, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore                 \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_compare_swap_nbi,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_fetch_inc(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_fetch_inc, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore              \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_fetch_inc,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_fetch_inc_nbi(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_fetch_inc_nbi, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore              \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_fetch_inc_nbi,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_inc(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_inc, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore        \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_inc,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_fetch_add(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_fetch_add, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore              \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_fetch_add,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_fetch_add_nbi(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_fetch_add_nbi, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore              \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_fetch_add_nbi,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_add(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_add, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore        \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_add,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

/* Deprecated APIs start */
#define shmem_cswap(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_compare_swap, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore   \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_compare_swap,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_finc(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_fetch_inc, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore  \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_fetch_inc,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore      \
    )(__VA_ARGS__)

#define shmem_inc(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_inc, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_inc,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore      \
    )(__VA_ARGS__)

#define shmem_fadd(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_fetch_add, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore  \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_fetch_add,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore      \
    )(__VA_ARGS__)

#define shmem_add(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_add, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_add,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore     \
    )(__VA_ARGS__)
/* Deprecated APIs end */
#endif /* OSHMPI_HAVE_C11 */
/* *INDENT-ON* */

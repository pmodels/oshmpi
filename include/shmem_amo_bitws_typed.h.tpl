/* The following lines are automatically generated. DO NOT EDIT. */
/* TPL_BLOCK_START */
TYPE shmem_TYPENAME_atomic_fetch_and(TYPE * dest, TYPE value, int pe);
TYPE shmem_ctx_TYPENAME_atomic_fetch_and(shmem_ctx_t ctx, TYPE * dest, TYPE value, int pe);
void shmem_TYPENAME_atomic_and(TYPE * dest, TYPE value, int pe);
void shmem_ctx_TYPENAME_atomic_and(shmem_ctx_t ctx, TYPE * dest, TYPE value, int pe);
TYPE shmem_TYPENAME_atomic_fetch_or(TYPE * dest, TYPE value, int pe);
TYPE shmem_ctx_TYPENAME_atomic_fetch_or(shmem_ctx_t ctx, TYPE * dest, TYPE value, int pe);
void shmem_TYPENAME_atomic_or(TYPE * dest, TYPE value, int pe);
void shmem_ctx_TYPENAME_atomic_or(shmem_ctx_t ctx, TYPE * dest, TYPE value, int pe);
TYPE shmem_TYPENAME_atomic_fetch_xor(TYPE * dest, TYPE value, int pe);
TYPE shmem_ctx_TYPENAME_atomic_fetch_xor(shmem_ctx_t ctx, TYPE * dest, TYPE value, int pe);
void shmem_TYPENAME_atomic_xor(TYPE * dest, TYPE value, int pe);
void shmem_ctx_TYPENAME_atomic_xor(shmem_ctx_t ctx, TYPE * dest, TYPE value, int pe);

void shmem_TYPENAME_atomic_fetch_and_nbi(TYPE * fetch, TYPE * dest, TYPE value, int pe);
void shmem_ctx_TYPENAME_atomic_fetch_and_nbi(shmem_ctx_t ctx, TYPE * fetch, TYPE * dest, TYPE value,
                                             int pe);
void shmem_TYPENAME_atomic_fetch_or_nbi(TYPE * fetch, TYPE * dest, TYPE value, int pe);
void shmem_ctx_TYPENAME_atomic_fetch_or_nbi(shmem_ctx_t ctx, TYPE * fetch, TYPE * dest, TYPE value,
                                            int pe);
void shmem_TYPENAME_atomic_fetch_xor_nbi(TYPE * fetch, TYPE * dest, TYPE value, int pe);
void shmem_ctx_TYPENAME_atomic_fetch_xor_nbi(shmem_ctx_t ctx, TYPE * fetch, TYPE * dest, TYPE value,
                                             int pe);
/* TPL_BLOCK_END */

/* *INDENT-OFF* */
#if OSHMPI_HAVE_C11
#define shmem_atomic_fetch_and(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_fetch_and, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore       \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_fetch_and,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_and(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_and, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_and,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_fetch_or(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_fetch_or, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore      \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_fetch_or,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_or(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_or, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore\
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_or,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_fetch_xor(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_fetch_xor, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore       \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_fetch_xor,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_xor(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_xor, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_xor,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_fetch_and_nbi(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_fetch_and_nbi, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore       \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_fetch_and_nbi,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_fetch_or_nbi(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_fetch_or_nbi, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore      \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_fetch_or_nbi,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#define shmem_atomic_fetch_xor_nbi(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_ctx_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_ctx_TYPENAME_atomic_fetch_xor_nbi, \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore       \
        ), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_atomic_fetch_xor_nbi,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore       \
    )(__VA_ARGS__)

#endif /* OSHMPI_HAVE_C11 */
/* *INDENT-ON* */

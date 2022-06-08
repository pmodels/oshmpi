/* The following lines are automatically generated. DO NOT EDIT. */
/* TPL_BLOCK_START */
void shmem_TYPENAME_wait_until(TYPE * ivar, int cmp, TYPE cmp_value);
void shmem_TYPENAME_wait_until_all(TYPE * ivars, size_t nelems, const int *status, int cmp,
                                   TYPE cmp_value);
size_t shmem_TYPENAME_wait_until_any(TYPE * ivars, size_t nelems, const int *status, int cmp,
                                     TYPE cmp_value);
size_t shmem_TYPENAME_wait_until_some(TYPE * ivars, size_t nelems, size_t *indices,
                                      const int *status, int cmp, TYPE cmp_value);
void shmem_TYPENAME_wait_until_all_vector(TYPE * ivars, size_t nelems, const int *status, int cmp,
                                          TYPE * cmp_values);
size_t shmem_TYPENAME_wait_until_any_vector(TYPE * ivars, size_t nelems, const int *status, int cmp,
                                            TYPE * cmp_values);
size_t shmem_TYPENAME_wait_until_some_vector(TYPE * ivars, size_t nelems, size_t *indices,
                                             const int *status, int cmp, TYPE * cmp_values);
int shmem_TYPENAME_test(TYPE * ivar, int cmp, TYPE cmp_value);
int shmem_TYPENAME_test_all(TYPE * ivars, size_t nelems, const int *status, int cmp,
                            TYPE cmp_value);
size_t shmem_TYPENAME_test_any(TYPE * ivars, size_t nelems, const int *status, int cmp,
                               TYPE cmp_value);
size_t shmem_TYPENAME_test_some(TYPE * ivars, size_t nelems, size_t *indices, const int *status,
                                int cmp, TYPE * cmp_values);
int shmem_TYPENAME_test_all_vector(TYPE * ivars, size_t nelems, const int *status, int cmp,
                                   TYPE * cmp_values);
size_t shmem_TYPENAME_test_any_vector(TYPE * ivars, size_t nelems, const int *status, int cmp,
                                      TYPE * cmp_values);
size_t shmem_TYPENAME_test_some_vector(TYPE * ivars, size_t nelems, size_t *indices,
                                       const int *status, int cmp, TYPE * cmp_values);
/* TPL_BLOCK_END */

/* *INDENT-OFF* */
#if OSHMPI_HAVE_C11
#define shmem_wait_until(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_wait_until,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore   \
    )(__VA_ARGS__)

#define shmem_wait_until_all(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_wait_until_all,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore   \
    )(__VA_ARGS__)

#define shmem_wait_until_any(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_wait_until_any,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore   \
    )(__VA_ARGS__)

#define shmem_wait_until_some(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_wait_until_some,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore   \
    )(__VA_ARGS__)

#define shmem_wait_until_all_vector(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_wait_until_all_vector,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore   \
    )(__VA_ARGS__)

#define shmem_wait_until_any_vector(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_wait_until_any_vector,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore   \
    )(__VA_ARGS__)

#define shmem_wait_until_some_vector(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_wait_until_some_vector,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore   \
    )(__VA_ARGS__)

#define shmem_test(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_test,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore   \
    )(__VA_ARGS__)

#define shmem_test_all(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_test_all,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore   \
    )(__VA_ARGS__)

#define shmem_test_any(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_test_any,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore   \
    )(__VA_ARGS__)

#define shmem_test_some(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_test_some,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore   \
    )(__VA_ARGS__)

#define shmem_test_all_vector(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_test_all_vector,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore   \
    )(__VA_ARGS__)

#define shmem_test_any_vector(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_test_any_vector,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore   \
    )(__VA_ARGS__)

#define shmem_test_some_vector(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_test_some_vector,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore   \
    )(__VA_ARGS__)
#endif /* OSHMPI_HAVE_C11 */
/* *INDENT-ON* */

/* The following lines are automatically generated. DO NOT EDIT. */
/* TPL_BLOCK_START */
void shmem_TYPENAME_wait_until(TYPE * ivar, int cmp, TYPE cmp_value);
int shmem_TYPENAME_test(TYPE * ivar, int cmp, TYPE cmp_value);
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

#define shmem_test(...)  \
    _Generic(OSHMPI_C11_CTX_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        /* TPL_C11_BLOCK_START */
        TYPE*: shmem_TYPENAME_test,     \
        /* TPL_C11_BLOCK_END */
        default: shmem_c11_type_ignore   \
    )(__VA_ARGS__)
#endif /* OSHMPI_HAVE_C11 */
/* *INDENT-ON* */
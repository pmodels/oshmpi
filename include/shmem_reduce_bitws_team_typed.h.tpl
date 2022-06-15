/* The following lines are automatically generated. DO NOT EDIT. */
/* TPL_BLOCK_START */
int shmem_TYPENAME_and_reduce(shmem_team_t team, TYPE * dest, const TYPE * source, int nreduce);
int shmem_TYPENAME_or_reduce(shmem_team_t team, TYPE * dest, const TYPE * source, int nreduce);
int shmem_TYPENAME_xor_reduce(shmem_team_t team, TYPE * dest, const TYPE * source, int nreduce);
/* TPL_BLOCK_END */

/* *INDENT-OFF* */
#if OSHMPI_HAVE_C11
#define shmem_and_reduce(...)  \
    _Generic(OSHMPI_C11_TEAM_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_team_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_TYPENAME_and_reduce,   \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore \
        ), \
        default: shmem_c11_type_ignore     \
    )(__VA_ARGS__)

#define shmem_or_reduce(...)  \
    _Generic(OSHMPI_C11_TEAM_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_team_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_TYPENAME_or_reduce,   \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore \
        ), \
        default: shmem_c11_type_ignore     \
    )(__VA_ARGS__)

#define shmem_xor_reduce(...)  \
    _Generic(OSHMPI_C11_TEAM_VAL(OSHMPI_C11_ARG0(__VA_ARGS__)), \
        shmem_team_t:  _Generic((OSHMPI_C11_ARG1(__VA_ARGS__)), \
/* TPL_C11_BLOCK_START */
            TYPE*: shmem_TYPENAME_xor_reduce,   \
/* TPL_C11_BLOCK_END */
            default: shmem_c11_type_ignore \
        ), \
        default: shmem_c11_type_ignore     \
    )(__VA_ARGS__)
#endif /* OSHMPI_HAVE_C11 */
/* *INDENT-ON* */

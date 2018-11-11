/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <shmem.h>
#include <unistd.h>
#include "oshmpi_impl.h"

#if defined(USE_LINUX)
/* http://www.salbut.net/public/gcc-pdf/ld.pdf */
#include <unistd.h>
extern char __data_start;
extern char _end;
#define OSHMPI_DATA_START (void *) &__data_start
#define OSHMPI_DATA_SIZE ((char *) &_end - (char *) &__data_start)
#elif defined(USE_OSX)
/* http:  //www.manpagez.com/man/3/get_etext/ */
#include <mach-o/getsect.h>
unsigned long get_end();
unsigned long get_etext();
#define OSHMPI_DATA_START (void*)get_etext()
#define OSHMPI_DATA_SIZE (get_end() - get_etext())
#elif defined(USE_FREEBSD)
/* https://www.freebsd.org/cgi/man.cgi?query=edata */
extern end;
extern etext;
extern edata;
#define OSHMPI_DATA_START (void*) (&etext)
#define OSHMPI_DATA_SIZE ((char *) (&end) - (char *) (&etext))
#else
#define OSHMPI_DATA_START 0
#define OSHMPI_DATA_SIZE 0
#endif

typedef struct OSHMPI_mpi_info_args {
    char accumulate_ops[MPI_MAX_INFO_VAL];
    char which_accumulate_ops[MPI_MAX_INFO_VAL];        /* MPICH specific */
} OSHMPI_mpi_info_args_t;

OSHMPI_global_t OSHMPI_global = { 0 };
OSHMPI_env_t OSHMPI_env = { 0 };

OSHMPI_STATIC_INLINE_PREFIX void initialize_symm_text(OSHMPI_mpi_info_args_t info_args)
{
    MPI_Info info = MPI_INFO_NULL;
    OSHMPI_global.symm_data_win = MPI_WIN_NULL;

    OSHMPI_global.symm_data_base = OSHMPI_DATA_START;
    OSHMPI_global.symm_data_size = (MPI_Aint) OSHMPI_DATA_SIZE;

    if (OSHMPI_global.symm_data_base == NULL || OSHMPI_global.symm_data_size == 0)
        OSHMPI_ERR_ABORT("Invalid data segment information: base %p, size 0x%lx\n",
                         OSHMPI_global.symm_data_base, OSHMPI_global.symm_data_size);

    OSHMPI_CALLMPI(MPI_Info_create(&info));
    OSHMPI_CALLMPI(MPI_Info_set(info, "accumulate_ops", (const char *) info_args.accumulate_ops));
    OSHMPI_CALLMPI(MPI_Info_set
                   (info, "which_accumulate_ops", (const char *) info_args.which_accumulate_ops));

    /* Allocate RMA window */
    OSHMPI_CALLMPI(MPI_Win_create
                   (OSHMPI_global.symm_data_base, (MPI_Aint) OSHMPI_global.symm_data_size,
                    1 /* disp_unit */ , info, OSHMPI_global.comm_world,
                    &OSHMPI_global.symm_data_win));

    OSHMPI_CALLMPI(MPI_Win_lock_all(MPI_MODE_NOCHECK, OSHMPI_global.symm_data_win));
    OSHMPI_CALLMPI(MPI_Info_free(&info));

    OSHMPI_DBGMSG("Initialized symm data at base %p, size 0x%lx.\n",
                  OSHMPI_global.symm_data_base, OSHMPI_global.symm_data_size);
}

OSHMPI_STATIC_INLINE_PREFIX void initialize_symm_heap(OSHMPI_mpi_info_args_t info_args)
{
    uint64_t symm_heap_size;
    MPI_Info info = MPI_INFO_NULL;
    size_t pagesize = (size_t) sysconf(_SC_PAGESIZE);

    OSHMPI_global.symm_heap_base = NULL;
    OSHMPI_global.symm_heap_mspace = NULL;
    OSHMPI_global.symm_heap_win = MPI_WIN_NULL;
    OSHMPI_global.symm_heap_size = OSHMPI_env.symm_heap_size;

    /* Ensure extra bookkeeping space in MSPACE */
    symm_heap_size = (uint64_t) OSHMPI_global.symm_heap_size + OSHMPI_DLMALLOC_MIN_MSPACE_SIZE;
    symm_heap_size = OSHMPI_ALIGN(symm_heap_size, pagesize);

    /* Allocate RMA window */
    OSHMPI_CALLMPI(MPI_Info_create(&info));
    OSHMPI_CALLMPI(MPI_Info_set(info, "alloc_shm", "true"));    /* MPICH specific */
    OSHMPI_CALLMPI(MPI_Info_set(info, "accumulate_ops", (const char *) info_args.accumulate_ops));
    OSHMPI_CALLMPI(MPI_Info_set
                   (info, "which_accumulate_ops", (const char *) info_args.which_accumulate_ops));

    OSHMPI_CALLMPI(MPI_Win_allocate((MPI_Aint) symm_heap_size, 1 /* disp_unit */ , info,
                                    OSHMPI_global.comm_world, &OSHMPI_global.symm_heap_base,
                                    &OSHMPI_global.symm_heap_win));
    OSHMPI_ASSERT(OSHMPI_global.symm_heap_base != NULL);

    /* Initialize MSPACE */
    OSHMPI_global.symm_heap_mspace = create_mspace_with_base(OSHMPI_global.symm_heap_base,
                                                             symm_heap_size,
                                                             OSHMPI_global.thread_level ==
                                                             SHMEM_THREAD_MULTIPLE ? 1 : 0);
    OSHMPI_ASSERT(OSHMPI_global.symm_heap_mspace != NULL);
    OSHMPI_THREAD_INIT_CS(&OSHMPI_global.symm_heap_mspace_cs);

    OSHMPI_CALLMPI(MPI_Win_lock_all(MPI_MODE_NOCHECK, OSHMPI_global.symm_heap_win));
    OSHMPI_CALLMPI(MPI_Info_free(&info));

    OSHMPI_DBGMSG("Initialized symm heap at base %p, size 0x%lx (allocated size 0x%lx).\n",
                  OSHMPI_global.symm_heap_base, OSHMPI_global.symm_heap_size, symm_heap_size);
}


OSHMPI_STATIC_INLINE_PREFIX void set_env_amo_ops(const char *str, uint32_t * ops_ptr)
{
    uint32_t ops = 0;
    char *value, *token, *savePtr = NULL;

    value = (char *) str;
    /* str can never be NULL. */
    OSHMPI_ASSERT(value);

    /* handle special value */
    if (!strncmp(value, "none", strlen("none"))) {
        *ops_ptr = 0;
        return;
    } else if (!strncmp(value, "any_op", strlen("any_op"))) {
        OSHMPI_amo_op_shift_t op_shift;
        /* add all ops */
        for (op_shift = 0; op_shift < OSHMPI_AMO_OP_LAST; op_shift++)
            ops |= (1 << op_shift);
        *ops_ptr = ops;
        return;
    }

    token = (char *) strtok_r(value, ",", &savePtr);
    while (token != NULL) {
        /* traverse op list (exclude null and last) and add the op if set */
        if (!strncmp(token, "cswap", strlen("cswap")))
            ops |= (1 << OSHMPI_AMO_CSWAP);
        else if (!strncmp(token, "finc", strlen("finc")))
            ops |= (1 << OSHMPI_AMO_FINC);
        else if (!strncmp(token, "inc", strlen("inc")))
            ops |= (1 << OSHMPI_AMO_INC);
        else if (!strncmp(token, "fadd", strlen("fadd")))
            ops |= (1 << OSHMPI_AMO_FADD);
        else if (!strncmp(token, "add", strlen("add")))
            ops |= (1 << OSHMPI_AMO_ADD);
        else if (!strncmp(token, "fetch", strlen("fetch")))
            ops |= (1 << OSHMPI_AMO_FETCH);
        else if (!strncmp(token, "set", strlen("set")))
            ops |= (1 << OSHMPI_AMO_SET);
        else if (!strncmp(token, "swap", strlen("swap")))
            ops |= (1 << OSHMPI_AMO_SWAP);
        else if (!strncmp(token, "fand", strlen("fand")))
            ops |= (1 << OSHMPI_AMO_FAND);
        else if (!strncmp(token, "and", strlen("and")))
            ops |= (1 << OSHMPI_AMO_AND);
        else if (!strncmp(token, "for", strlen("for")))
            ops |= (1 << OSHMPI_AMO_FOR);
        else if (!strncmp(token, "or", strlen("or")))
            ops |= (1 << OSHMPI_AMO_OR);
        else if (!strncmp(token, "fxor", strlen("fxor")))
            ops |= (1 << OSHMPI_AMO_FXOR);
        else if (!strncmp(token, "xor", strlen("xor")))
            ops |= (1 << OSHMPI_AMO_XOR);

        token = (char *) strtok_r(NULL, ",", &savePtr);
    }

    /* update info only when any valid value is set */
    if (ops)
        *ops_ptr = ops;
}

OSHMPI_STATIC_INLINE_PREFIX void getstr_env_amo_ops(uint32_t val, char *buf, size_t maxlen)
{
    int c = 0;

    OSHMPI_ASSERT(maxlen >= strlen("cswap,finc,inc,fadd,add,fetch,set,swap,"
                                   "fadd,and,for,or,fxor,xor") + 1);

    if (val & (1 << OSHMPI_AMO_CSWAP))
        c += snprintf(buf + c, maxlen - c, "cswap");
    if (val & (1 << OSHMPI_AMO_FINC))
        c += snprintf(buf + c, maxlen - c, "%sfinc", (c > 0) ? "," : "");
    if (val & (1 << OSHMPI_AMO_INC))
        c += snprintf(buf + c, maxlen - c, "%sinc", (c > 0) ? "," : "");
    if (val & (1 << OSHMPI_AMO_FADD))
        c += snprintf(buf + c, maxlen - c, "%sfadd", (c > 0) ? "," : "");
    if (val & (1 << OSHMPI_AMO_ADD))
        c += snprintf(buf + c, maxlen - c, "%sadd", (c > 0) ? "," : "");
    if (val & (1 << OSHMPI_AMO_FETCH))
        c += snprintf(buf + c, maxlen - c, "%sfetch", (c > 0) ? "," : "");
    if (val & (1 << OSHMPI_AMO_SET))
        c += snprintf(buf + c, maxlen - c, "%sset", (c > 0) ? "," : "");
    if (val & (1 << OSHMPI_AMO_SWAP))
        c += snprintf(buf + c, maxlen - c, "%sswap", (c > 0) ? "," : "");
    if (val & (1 << OSHMPI_AMO_FAND))
        c += snprintf(buf + c, maxlen - c, "%sfand", (c > 0) ? "," : "");
    if (val & (1 << OSHMPI_AMO_AND))
        c += snprintf(buf + c, maxlen - c, "%sand", (c > 0) ? "," : "");
    if (val & (1 << OSHMPI_AMO_FOR))
        c += snprintf(buf + c, maxlen - c, "%sfor", (c > 0) ? "," : "");
    if (val & (1 << OSHMPI_AMO_OR))
        c += snprintf(buf + c, maxlen - c, "%sor", (c > 0) ? "," : "");
    if (val & (1 << OSHMPI_AMO_FXOR))
        c += snprintf(buf + c, maxlen - c, "%sfxor", (c > 0) ? "," : "");
    if (val & (1 << OSHMPI_AMO_XOR))
        c += snprintf(buf + c, maxlen - c, "%sxor", (c > 0) ? "," : "");

    if (c == 0)
        strncpy(buf, "none", maxlen);
}

OSHMPI_STATIC_INLINE_PREFIX void print_env(void)
{
    if ((OSHMPI_env.info || OSHMPI_env.verbose) && OSHMPI_global.world_rank == 0)
        OSHMPI_PRINTF("SHMEM environment variables:\n"
                      "    SHMEM_SYMMETRIC_SIZE %ld (bytes)\n"
                      "    SHMEM_DEBUG          %d (Invalid if OSHMPI is built with --enable-fast)\n"
                      "    SHMEM_VERSION        %d\n"
                      "    SHMEM_INFO           %d\n\n",
                      OSHMPI_env.symm_heap_size, OSHMPI_env.debug,
                      OSHMPI_env.version, OSHMPI_env.info);

    /* *INDENT-OFF* */
    if (OSHMPI_env.verbose && OSHMPI_global.world_rank == 0) {
        char amo_ops_str[256];
        OSHMPI_PRINTF("OSHMPI configuration:\n"
                      "    --enable-fast                "
#ifdef OSHMPI_ENABLE_FAST
                      "yes\n"
#else
                      "no\n"
#endif
                      "    --enable-threads             "
#ifdef OSHMPI_ENABLE_THREAD_SINGLE
                      "single\n"
#elif defined(OSHMPI_ENABLE_THREAD_FUNNELED)
                      "funneled\n"
#elif defined(OSHMPI_ENABLE_THREAD_SERIALIZED)
                      "serialized\n"
#else
                      "multiple\n"
#endif
                      "    --enable-amo                 "
#ifdef OSHMPI_ENABLE_DIRECT_AMO
                      "direct\n"
#elif defined(OSHMPI_ENABLE_AM_AMO)
                      "am\n"
#else
                      "auto\n"
#endif
                      "    --enable-async-thread        "
#ifdef OSHMPI_ENABLE_AMO_ASYNC_THREAD
                      "yes\n"
#elif defined(OSHMPI_RUNTIME_AMO_ASYNC_THREAD)
                      "runtime\n"
#else
                      "no\n"
#endif
                      "\n");

        getstr_env_amo_ops(OSHMPI_env.amo_ops, amo_ops_str, sizeof(amo_ops_str));

        OSHMPI_PRINTF("OSHMPI environment variables:\n"
                      "    OSHMPI_VERBOSE               %d\n"
                      "    OSHMPI_AMO_OPS               %s\n"
                      "    OSHMPI_ENABLE_ASYNC_THREAD   %d\n\n",
                      OSHMPI_env.verbose, amo_ops_str,
                      OSHMPI_env.enable_async_thread);
    }
    /* *INDENT-ON* */
}

OSHMPI_STATIC_INLINE_PREFIX void initialize_env(void)
{
    char *val = NULL;

    /* Number of bytes to allocate for symmetric heap. */
    OSHMPI_env.symm_heap_size = OSHMPI_DEFAULT_SYMM_HEAP_SIZE;
    val = getenv("SHMEM_SYMMETRIC_SIZE");
    if (val && strlen(val))
        OSHMPI_env.symm_heap_size = (MPI_Aint) OSHMPIU_str_to_size(val);
    if (OSHMPI_env.symm_heap_size < 0)
        OSHMPI_ERR_ABORT("Invalid SHMEM_SYMMETRIC_SIZE: %ld\n", OSHMPI_env.symm_heap_size);

    /* FIXME: determine system available memory size */

    /* Debug message. Any non-zero value will enable it. */
    OSHMPI_env.debug = 0;
    val = getenv("SHMEM_DEBUG");
    if (val && strlen(val))
        OSHMPI_env.debug = atoi(val);
    if (OSHMPI_env.debug != 0)
        OSHMPI_env.debug = 1;

    /* Print the library version at start-up. */
    OSHMPI_env.version = 0;
    val = getenv("SHMEM_VERSION");
    if (val && strlen(val))
        OSHMPI_env.version = atoi(val);
    if (OSHMPI_env.version != 0)
        OSHMPI_env.version = 1;

    /* Print helpful text about environment variables. */
    OSHMPI_env.info = 0;
    val = getenv("SHMEM_INFO");
    if (val && strlen(val))
        OSHMPI_env.info = atoi(val);
    if (OSHMPI_env.info != 0)
        OSHMPI_env.info = 1;

    /* Print OSHMPI environment variables including standard and extensions. */
    OSHMPI_env.verbose = 0;
    val = getenv("OSHMPI_VERBOSE");
    if (val && strlen(val))
        OSHMPI_env.verbose = atoi(val);
    if (OSHMPI_env.verbose != 0)
        OSHMPI_env.verbose = 1;

    OSHMPI_env.amo_ops = 0;
    val = getenv("OSHMPI_AMO_OPS");
    if (val && strlen(val))
        set_env_amo_ops(val, &OSHMPI_env.amo_ops);
    else
        set_env_amo_ops("any_op", &OSHMPI_env.amo_ops); /* default */

#ifdef OSHMPI_ENABLE_AMO_ASYNC_THREAD
    OSHMPI_env.enable_async_thread = 1;
#elif defined(OSHMPI_RUNTIME_AMO_ASYNC_THREAD)
    OSHMPI_env.enable_async_thread = 0;
    val = getenv("OSHMPI_ENABLE_ASYNC_THREAD");
    if (val && strlen(val))
        OSHMPI_env.enable_async_thread = atoi(val);
    if (OSHMPI_env.enable_async_thread != 0)
        OSHMPI_env.enable_async_thread = 1;
#else
    OSHMPI_env.enable_async_thread = 0;
#endif
}

OSHMPI_STATIC_INLINE_PREFIX void set_mpi_info_args(OSHMPI_mpi_info_args_t * info)
{
    int c = 0;
    size_t maxlen = MPI_MAX_INFO_VAL;
    unsigned int nops = 0;
    uint32_t val = OSHMPI_env.amo_ops;

    memset(info->accumulate_ops, 0, sizeof(info->accumulate_ops));
    memset(info->which_accumulate_ops, 0, sizeof(info->which_accumulate_ops));

    /* Info key which_accumulate_ops is MPICH specific, valid values include:
     * max,min,sum,prod,maxloc,minloc,band,bor,bxor,land,lor,lxor,replace,no_op,cswap */

    OSHMPI_ASSERT(MPI_MAX_INFO_VAL >= strlen("cswap,sum,band,bor,bxor,no_op,replace") + 1);

    if (val & (1 << OSHMPI_AMO_CSWAP)) {
        c += snprintf(info->which_accumulate_ops + c, maxlen - c, "cswap");
        nops++;
    }
    if ((val & (1 << OSHMPI_AMO_FINC)) || (val & (1 << OSHMPI_AMO_INC)) ||
        (val & (1 << OSHMPI_AMO_FADD)) || (val & (1 << OSHMPI_AMO_ADD))) {
        c += snprintf(info->which_accumulate_ops + c, maxlen - c, "%ssum", (c > 0) ? "," : "");
        nops++;
    }
    if (val & (1 << OSHMPI_AMO_FETCH)) {
        c += snprintf(info->which_accumulate_ops + c, maxlen - c, "%sno_op", (c > 0) ? "," : "");
        nops++;
    }
    if ((val & (1 << OSHMPI_AMO_SET)) || (val & (1 << OSHMPI_AMO_SWAP))) {
        c += snprintf(info->which_accumulate_ops + c, maxlen - c, "%sreplace", (c > 0) ? "," : "");
        nops++;
    }
    if ((val & (1 << OSHMPI_AMO_FAND)) || (val & (1 << OSHMPI_AMO_AND))) {
        c += snprintf(info->which_accumulate_ops + c, maxlen - c, "%sband", (c > 0) ? "," : "");
        nops++;
    }
    if ((val & (1 << OSHMPI_AMO_FOR)) || (val & (1 << OSHMPI_AMO_OR))) {
        c += snprintf(info->which_accumulate_ops + c, maxlen - c, "%sbor", (c > 0) ? "," : "");
        nops++;
    }
    if ((val & (1 << OSHMPI_AMO_FXOR)) || (val & (1 << OSHMPI_AMO_XOR))) {
        c += snprintf(info->which_accumulate_ops + c, maxlen - c, "%sbxor", (c > 0) ? "," : "");
        nops++;
    }

    if (c == 0)
        strncpy(info->which_accumulate_ops, "none", maxlen);

    /* With MPI standard info values same_op or same_op_no_op,
     * we can enable MPI accumulate based atomics. MPI-3 is required at configure. */
    maxlen = MPI_MAX_INFO_VAL;  /* reset */
    if (nops == 1) {
        strncpy(info->accumulate_ops, "same_op", maxlen);
        OSHMPI_global.amo_direct = 1;
    } else if (nops == 2 && (val & (1 << OSHMPI_AMO_FETCH))) {
        strncpy(info->accumulate_ops, "same_op_no_op", maxlen);
        OSHMPI_global.amo_direct = 1;
    } else      /* MPI default */
        strncpy(info->accumulate_ops, "same_op_no_op", maxlen);
}

int OSHMPI_initialize_thread(int required, int *provided)
{
    int mpi_errno = MPI_SUCCESS;
    int mpi_provided = 0, mpi_initialized = 0, mpi_required = 0;
    OSHMPI_mpi_info_args_t info_args;

    if (OSHMPI_global.is_initialized)
        goto fn_exit;

    initialize_env();

    if (required != SHMEM_THREAD_SINGLE && required != SHMEM_THREAD_FUNNELED
        && required != SHMEM_THREAD_SERIALIZED && required != SHMEM_THREAD_MULTIPLE)
        OSHMPI_ERR_ABORT("Unknown OpenSHMEM thread support level: %d\n", required);

    if (required > OSHMPI_DEFAULT_THREAD_SAFETY)
        OSHMPI_ERR_ABORT("OpenSHMEM thread level %s is not enabled. "
                         "Upgrade --enable-threads option at configure.\n",
                         OSHMPI_thread_level_str(required));

    OSHMPI_CALLMPI(MPI_Initialized(&mpi_initialized));
    if (mpi_initialized) {
        /* If MPI has already be initialized, we only query the thread safety. */
        OSHMPI_CALLMPI(MPI_Query_thread(&mpi_provided));
    } else {
        /* Initialize MPI */
        mpi_required = required;

        /* Force thread multiple when async thread is enabled. */
#ifdef OSHMPI_ENABLE_AMO_ASYNC_THREAD
        mpi_required = MPI_THREAD_MULTIPLE;
#elif defined(OSHMPI_RUNTIME_AMO_ASYNC_THREAD)
        if (OSHMPI_env.enable_async_thread)
            mpi_required = MPI_THREAD_MULTIPLE;
#endif

        OSHMPI_CALLMPI(MPI_Init_thread(NULL, NULL, mpi_required, &mpi_provided));
    }

    /* Abort if provided safety is lower than the requested one.
     * MPI_THREAD_SINGLE < MPI_THREAD_FUNNELED < MPI_THREAD_SERIALIZED < MPI_THREAD_MULTIPLE */
    if (mpi_provided < required) {
        OSHMPI_ERR_ABORT("The MPI library does not support the required thread support:"
                         "required: %s, provided: %s.\n",
                         OSHMPI_thread_level_str(required), OSHMPI_thread_level_str(mpi_provided));
    }

    /* OSHMPI internal routines are protected only when user explicitly requires multiple
     * safety, thus we do not expose the actual safety provided by MPI if it is higher. */
    OSHMPI_global.thread_level = required;

    /* Duplicate comm world for oshmpi use. */
    OSHMPI_CALLMPI(MPI_Comm_dup(MPI_COMM_WORLD, &OSHMPI_global.comm_world));
    OSHMPI_CALLMPI(MPI_Comm_size(OSHMPI_global.comm_world, &OSHMPI_global.world_size));
    OSHMPI_CALLMPI(MPI_Comm_rank(OSHMPI_global.comm_world, &OSHMPI_global.world_rank));
    OSHMPI_CALLMPI(MPI_Comm_group(OSHMPI_global.comm_world, &OSHMPI_global.comm_world_group));

    print_env();

    set_mpi_info_args(&info_args);

    initialize_symm_text(info_args);

    initialize_symm_heap(info_args);

    OSHMPI_coll_initialize();
    OSHMPI_amo_initialize();

    OSHMPI_am_progress_mpi_barrier(OSHMPI_global.comm_world);
    OSHMPI_global.is_initialized = 1;

  fn_exit:
    if (provided)
        *provided = OSHMPI_global.thread_level;
    return mpi_errno;
}

OSHMPI_STATIC_INLINE_PREFIX int finalize_impl(void)
{
    int mpi_errno = MPI_SUCCESS;
    int mpi_finalized = 0;

    OSHMPI_CALLMPI(MPI_Finalized(&mpi_finalized));
    if (mpi_finalized)
        OSHMPI_ERR_ABORT("The MPI library has already been finalized, "
                         "OSHMPI_finalize cannot complete.\n");

    /* Implicit global barrier is required to ensure
     * that pending communications are completed and that no resources
     * are released until all PEs have entered shmem_finalize.
     * The completion part is ensured in unlock calls.*/
    OSHMPI_am_progress_mpi_barrier(OSHMPI_global.comm_world);

    OSHMPI_coll_finalize();
    OSHMPI_amo_finalize();

    if (OSHMPI_global.symm_heap_win != MPI_WIN_NULL) {
        OSHMPI_CALLMPI(MPI_Win_unlock_all(OSHMPI_global.symm_heap_win));
        OSHMPI_CALLMPI(MPI_Win_free(&OSHMPI_global.symm_heap_win));
    }
    OSHMPI_THREAD_DESTROY_CS(&OSHMPI_global.symm_heap_mspace_cs);

    if (OSHMPI_global.symm_data_win != MPI_WIN_NULL) {
        OSHMPI_CALLMPI(MPI_Win_unlock_all(OSHMPI_global.symm_data_win));
        OSHMPI_CALLMPI(MPI_Win_free(&OSHMPI_global.symm_data_win));
    }

    OSHMPI_global.is_initialized = 0;

    OSHMPI_CALLMPI(MPI_Group_free(&OSHMPI_global.comm_world_group));
    OSHMPI_CALLMPI(MPI_Comm_free(&OSHMPI_global.comm_world));
    OSHMPI_CALLMPI(MPI_Finalize());

    return mpi_errno;
}

/* Implicitly called at program exit, valid only when program is initialized
 * by start_pes and the finalize call is not explicitly called. */
void OSHMPI_implicit_finalize(void)
{
    if (OSHMPI_global.is_start_pes_initialized && OSHMPI_global.is_initialized)
        finalize_impl();
}

int OSHMPI_finalize(void)
{
    int mpi_errno = MPI_SUCCESS;

    /* Skip if a finalize is already called or the program is not
     * initialized yet. */
    if (OSHMPI_global.is_initialized)
        mpi_errno = finalize_impl();

    OSHMPI_DBGMSG("finalized ---\n");
    return mpi_errno;
}

void OSHMPI_global_exit(int status)
{
    OSHMPI_DBGMSG("status %d !!!\n", status);

    /* Force termination of an entire program. Make it non-stop 
     * to avoid a c11 warning about noreturn. */
    do {
        MPI_Abort(OSHMPI_global.comm_world, status);
    } while (1);
}

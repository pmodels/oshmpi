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

OSHMPI_global_t OSHMPI_global = { 0 };
OSHMPI_env_t OSHMPI_env = { 0 };

void OSHMPI_set_mpi_info_args(MPI_Info info)
{
    unsigned int nops;

    const char *amo_std_types =
        "int:1,long:1,longlong:1,uint:1,ulong:1,ulonglong:1,int32:1,int64:1,uint32:1,uint64:1";
    const char *amo_ext_types =
        "float:1,double:1,int:1,long:1,longlong:1,uint:1,ulong:1,ulonglong:1,int32:1,int64:1,uint32:1,uint64:1";
    const char *amo_bitws_types = "uint:1,ulong:1,ulonglong:1,int32:1,int64:1,uint32:1,uint64:1";

    OSHMPI_ASSERT(MPI_MAX_INFO_VAL >= strlen("cswap,sum,band,bor,bxor,no_op,replace") + 1);

    nops = 0;
    if (OSHMPI_env.amo_ops & (1 << OSHMPI_AMO_CSWAP)) {
        OSHMPI_CALLMPI(MPI_Info_set(info, "accumulate_op_types:cswap", amo_std_types));
        nops++;
    }
    if ((OSHMPI_env.amo_ops & (1 << OSHMPI_AMO_FINC)) || (OSHMPI_env.amo_ops) ||
        (OSHMPI_env.amo_ops & (1 << OSHMPI_AMO_FADD)) ||
        (OSHMPI_env.amo_ops & (1 << OSHMPI_AMO_ADD))) {
        OSHMPI_CALLMPI(MPI_Info_set(info, "accumulate_op_types:sum", amo_std_types));
        nops++;
    }
    if (OSHMPI_env.amo_ops & (1 << OSHMPI_AMO_FETCH)) {
        OSHMPI_CALLMPI(MPI_Info_set(info, "accumulate_op_types:no_op", amo_ext_types));
        nops++;
    }
    if ((OSHMPI_env.amo_ops & (1 << OSHMPI_AMO_SET)) ||
        (OSHMPI_env.amo_ops & (1 << OSHMPI_AMO_SWAP))) {
        OSHMPI_CALLMPI(MPI_Info_set(info, "accumulate_op_types:replace", amo_ext_types));
        nops++;
    }
    if ((OSHMPI_env.amo_ops & (1 << OSHMPI_AMO_FAND)) ||
        (OSHMPI_env.amo_ops & (1 << OSHMPI_AMO_AND))) {
        OSHMPI_CALLMPI(MPI_Info_set(info, "accumulate_op_types:band", amo_bitws_types));
        nops++;
    }
    if ((OSHMPI_env.amo_ops & (1 << OSHMPI_AMO_FOR)) || (OSHMPI_env.amo_ops & (1 << OSHMPI_AMO_OR))) {
        OSHMPI_CALLMPI(MPI_Info_set(info, "accumulate_op_types:bor", amo_bitws_types));
        nops++;
    }
    if ((OSHMPI_env.amo_ops & (1 << OSHMPI_AMO_FXOR)) ||
        (OSHMPI_env.amo_ops & (1 << OSHMPI_AMO_XOR))) {
        OSHMPI_CALLMPI(MPI_Info_set(info, "accumulate_op_types:bxor", amo_bitws_types));
        nops++;
    }

    /* accumulate_ops.
     * With MPI standard info values same_op or same_op_no_op,
     * we can enable MPI accumulate based atomics. MPI-3 is required at configure. */
    if (nops == 1) {
        OSHMPI_CALLMPI(MPI_Info_set(info, "accumulate_ops", "same_op"));
        OSHMPI_global.amo_direct = 1;
    } else if (nops == 2 && (OSHMPI_env.amo_ops & (1 << OSHMPI_AMO_FETCH))) {
        OSHMPI_CALLMPI(MPI_Info_set(info, "accumulate_ops", "same_op_no_op"));
        OSHMPI_global.amo_direct = 1;
    } else      /* MPI default */
        OSHMPI_CALLMPI(MPI_Info_set(info, "accumulate_ops", "same_op_no_op"));

    OSHMPI_CALLMPI(MPI_Info_set(info, "which_rma_ops", "put,get"));
    OSHMPI_CALLMPI(MPI_Info_set(info, "rma_op_types:put", "contig:unlimited,vector:unlimited"));
    OSHMPI_CALLMPI(MPI_Info_set(info, "rma_op_types:get", "contig:unlimited,vector:unlimited"));
#ifdef OSHMPI_USE_MPIX_RMA_ABS
    OSHMPI_CALLMPI(MPI_Info_set(info, "rma_abs", "true"));
#endif
}

#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
static void initialize_symm_win()
{
    MPI_Info info = MPI_INFO_NULL;
    OSHMPI_global.symm_ictx.win = MPI_WIN_NULL;
    OSHMPI_global.symm_ictx.outstanding_op = 0;
    OSHMPI_global.symm_ictx.disp_mode = OSHMPI_ABS_DISP;

    OSHMPI_CALLMPI(MPI_Info_create(&info));

    OSHMPI_set_mpi_info_args(info);
    OSHMPI_CALLMPI(MPI_Info_set(info, "coll_attach", "true"));

    /* Allocate RMA window */
    OSHMPI_CALLMPI(MPI_Win_create_dynamic
                   (info, OSHMPI_global.comm_world, &OSHMPI_global.symm_ictx.win));

    OSHMPI_CALLMPI(MPI_Win_lock_all(MPI_MODE_NOCHECK, OSHMPI_global.symm_ictx.win));
    OSHMPI_CALLMPI(MPI_Info_free(&info));

    OSHMPI_DBGMSG("Initialized symm window 0x%x.\n", OSHMPI_global.symm_ictx.win);
}

static void attach_symm_text(void)
{
    void *base = OSHMPI_DATA_START;
    MPI_Aint size = (MPI_Aint) OSHMPI_DATA_SIZE;

    if (base == NULL || size == 0)
        OSHMPI_ERR_ABORT("Invalid data segment information: base %p, size 0x%lx\n", base, size);

    /* Setup sobj attributes */
    int symm_flag = 0;
    OSHMPI_sobj_init_attr(&OSHMPI_global.symm_data_attr, SHMEMX_MEM_HOST, base, size);
    OSHMPI_sobj_symm_info_allgather(&OSHMPI_global.symm_data_attr, &symm_flag);
    OSHMPI_sobj_set_handle(&OSHMPI_global.symm_data_attr, OSHMPI_SOBJ_SYMM_DATA, symm_flag, 0);

    OSHMPI_CALLMPI(MPI_Win_attach(OSHMPI_global.symm_ictx.win, base, size));
    OSHMPI_CALLMPI(MPI_Barrier(OSHMPI_global.comm_world));

    OSHMPI_DBGMSG("Attached symm data at base %p, size 0x%lx, symm_flag %d, handle 0x%x\n",
                  OSHMPI_global.symm_data_attr.base, OSHMPI_global.symm_data_attr.size,
                  OSHMPI_SOBJ_HANDLE_GET_SYMMBIT(OSHMPI_global.symm_data_attr.handle),
                  OSHMPI_global.symm_data_attr.handle);
}

static void attach_symm_heap(void)
{
    uint64_t size, true_size;
    void *base = NULL;

    OSHMPI_global.symm_heap_mspace = NULL;

    /* Ensure extra bookkeeping space in MSPACE */
    size = OSHMPI_ALIGN(OSHMPI_env.symm_heap_size, OSHMPI_global.page_sz);
    true_size = size + OSHMPI_DLMALLOC_MIN_MSPACE_SIZE;
    true_size = OSHMPI_ALIGN(true_size, OSHMPI_global.page_sz);
    OSHMPI_global.symm_heap_true_size = true_size;

    /* Try to allocate symmetric heap. If fails, allocate separate heap
     * and check if the start address is the same. */
    if (OSHMPIU_allocate_symm_mem(true_size, &base))
        base = OSHMPIU_malloc(true_size);
    OSHMPI_ASSERT(base);

    /* Setup sobj attributes */
    int symm_flag = 0;
    OSHMPI_sobj_init_attr(&OSHMPI_global.symm_heap_attr, SHMEMX_MEM_HOST, base, size);
    OSHMPI_sobj_symm_info_allgather(&OSHMPI_global.symm_heap_attr, &symm_flag);
    OSHMPI_sobj_set_handle(&OSHMPI_global.symm_heap_attr, OSHMPI_SOBJ_SYMM_HEAP, symm_flag, 0);

    /* Initialize MSPACE */
    OSHMPI_global.symm_heap_mspace = create_mspace_with_base(base, true_size,
                                                             OSHMPI_global.thread_level ==
                                                             SHMEM_THREAD_MULTIPLE ? 1 : 0);
    OSHMPI_ASSERT(OSHMPI_global.symm_heap_mspace != NULL);
    OSHMPI_THREAD_INIT_CS(&OSHMPI_global.symm_heap_mspace_cs);

    OSHMPI_CALLMPI(MPI_Win_attach(OSHMPI_global.symm_ictx.win, base, true_size));
    OSHMPI_CALLMPI(MPI_Barrier(OSHMPI_global.comm_world));

    OSHMPI_DBGMSG
        ("Attached symm heap at base %p, size 0x%lx (allocated 0x%lx), symm_flag %d, handle 0x%x.\n",
         OSHMPI_global.symm_heap_attr.base, OSHMPI_global.symm_heap_attr.size,
         OSHMPI_global.symm_heap_true_size,
         OSHMPI_SOBJ_HANDLE_GET_SYMMBIT(OSHMPI_global.symm_heap_attr.handle),
         OSHMPI_global.symm_heap_attr.handle);
}
#else /* OSHMPI_ENABLE_DYNAMIC_WIN */

static void initialize_symm_text(void)
{
    MPI_Info info = MPI_INFO_NULL;
    OSHMPI_global.symm_data_ictx.win = MPI_WIN_NULL;
    OSHMPI_global.symm_data_ictx.outstanding_op = 0;

    void *base = OSHMPI_DATA_START;
    MPI_Aint size = (MPI_Aint) OSHMPI_DATA_SIZE;

    if (base == NULL || size == 0)
        OSHMPI_ERR_ABORT("Invalid data segment information: base %p, size 0x%lx\n", base, size);

    /* Setup sobj attributes */
    int symm_flag = 0;
    OSHMPI_sobj_init_attr(&OSHMPI_global.symm_data_attr, SHMEMX_MEM_HOST, base, size);
    OSHMPI_sobj_symm_info_allgather(&OSHMPI_global.symm_data_attr, &symm_flag);
    OSHMPI_sobj_set_handle(&OSHMPI_global.symm_data_attr, OSHMPI_SOBJ_SYMM_DATA, symm_flag, 0);

    OSHMPI_CALLMPI(MPI_Info_create(&info));
    OSHMPI_set_mpi_info_args(info);

    /* Allocate RMA window */
    OSHMPI_CALLMPI(MPI_Win_create(base, size, 1 /* disp_unit */ , info, OSHMPI_global.comm_world,
                                  &OSHMPI_global.symm_data_ictx.win));

    OSHMPI_CALLMPI(MPI_Win_lock_all(MPI_MODE_NOCHECK, OSHMPI_global.symm_data_ictx.win));
    OSHMPI_CALLMPI(MPI_Info_free(&info));

    OSHMPI_DBGMSG("Initialized symm data at base %p, size 0x%lx, symm_flag %d, handle 0x%x\n",
                  OSHMPI_global.symm_data_attr.base, OSHMPI_global.symm_data_attr.size,
                  OSHMPI_SOBJ_HANDLE_GET_SYMMBIT(OSHMPI_global.symm_data_attr.handle),
                  OSHMPI_global.symm_data_attr.handle);
}

static void initialize_symm_heap(void)
{
    uint64_t size, true_size;
    void *base;
    MPI_Info info = MPI_INFO_NULL;

    OSHMPI_global.symm_heap_mspace = NULL;
    OSHMPI_global.symm_heap_ictx.win = MPI_WIN_NULL;
    OSHMPI_global.symm_heap_ictx.outstanding_op = 0;

    /* Ensure extra bookkeeping space in MSPACE */
    size = OSHMPI_ALIGN(OSHMPI_env.symm_heap_size, OSHMPI_global.page_sz);
    true_size = size + OSHMPI_DLMALLOC_MIN_MSPACE_SIZE;
    true_size = OSHMPI_ALIGN(true_size, OSHMPI_global.page_sz);
    OSHMPI_global.symm_heap_true_size = true_size;

    /* Allocate RMA window */
    OSHMPI_CALLMPI(MPI_Info_create(&info));
    OSHMPI_CALLMPI(MPI_Info_set(info, "alloc_shm", "true"));    /* MPICH specific */
    OSHMPI_set_mpi_info_args(info);

    OSHMPI_CALLMPI(MPI_Win_allocate((MPI_Aint) true_size, 1 /* disp_unit */ , info,
                                    OSHMPI_global.comm_world, &base,
                                    &OSHMPI_global.symm_heap_ictx.win));
    OSHMPI_ASSERT(base);

    /* Setup sobj attributes */
    int symm_flag = 0;
    OSHMPI_sobj_init_attr(&OSHMPI_global.symm_heap_attr, SHMEMX_MEM_HOST, base, size);
    OSHMPI_sobj_symm_info_allgather(&OSHMPI_global.symm_heap_attr, &symm_flag);
    OSHMPI_sobj_set_handle(&OSHMPI_global.symm_heap_attr, OSHMPI_SOBJ_SYMM_HEAP, symm_flag, 0);

    /* Initialize MSPACE */
    OSHMPI_global.symm_heap_mspace = create_mspace_with_base(base, true_size,
                                                             OSHMPI_global.thread_level ==
                                                             SHMEM_THREAD_MULTIPLE ? 1 : 0);
    OSHMPI_ASSERT(OSHMPI_global.symm_heap_mspace != NULL);
    OSHMPI_THREAD_INIT_CS(&OSHMPI_global.symm_heap_mspace_cs);

    OSHMPI_CALLMPI(MPI_Win_lock_all(MPI_MODE_NOCHECK, OSHMPI_global.symm_heap_ictx.win));
    OSHMPI_CALLMPI(MPI_Info_free(&info));

    OSHMPI_DBGMSG
        ("Initialized symm heap at base %p, size 0x%lx (allocated 0x%lx), symm_flag %d, handle 0x%x.\n",
         OSHMPI_global.symm_heap_attr.base, OSHMPI_global.symm_heap_attr.size,
         OSHMPI_global.symm_heap_true_size,
         OSHMPI_SOBJ_HANDLE_GET_SYMMBIT(OSHMPI_global.symm_heap_attr.handle),
         OSHMPI_global.symm_heap_attr.handle);
}

#endif /* end of OSHMPI_ENABLE_AM_ASYNC_THREAD */

static void set_env_amo_ops(const char *str, uint32_t * ops_ptr)
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
        for (op_shift = OSHMPI_AMO_CSWAP; op_shift < OSHMPI_AMO_OP_LAST; op_shift++)
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

static void getstr_env_amo_ops(uint32_t val, char *buf, size_t maxlen)
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

#ifndef OSHMPI_ENABLE_FAST
static void set_env_dbg_mode(const char *str, OSHMPI_dbg_mode_t * dbg_mode)
{
    if (!strncmp(str, "am", strlen("am"))) {
        *dbg_mode = OSHMPI_DBG_AM;
    } else if (!strncmp(str, "direct", strlen("direct"))) {
        *dbg_mode = OSHMPI_DBG_DIRECT;
    }
}

static const char *getstr_env_dbg_mode(OSHMPI_dbg_mode_t dbg_mode)
{
    switch (dbg_mode) {
        case OSHMPI_DBG_AM:
            return "am";
        case OSHMPI_DBG_DIRECT:
            return "direct";
        default:
            return "auto";
    }
}
#endif

static void set_env_mpi_gpu_features(const char *str, uint32_t * features_ptr)
{
    uint32_t features = 0;
    char *value, *token, *savePtr = NULL;

    value = (char *) str;
    /* str can never be NULL. */
    OSHMPI_ASSERT(value);

    /* handle special value */
    if (!strncmp(value, "none", strlen("none"))) {
        *features_ptr = 0;
        return;
    } else if (!strncmp(value, "all", strlen("all"))) {
        *features_ptr = OSHMPI_SET_MPI_GPU_FEATURE(OSHMPI_MPI_GPU_PT2PT) |
            OSHMPI_SET_MPI_GPU_FEATURE(OSHMPI_MPI_GPU_PUT) |
            OSHMPI_SET_MPI_GPU_FEATURE(OSHMPI_MPI_GPU_GET) |
            OSHMPI_SET_MPI_GPU_FEATURE(OSHMPI_MPI_GPU_ACCUMULATES);
        return;
    }

    token = (char *) strtok_r(value, ",", &savePtr);
    while (token != NULL) {
        if (!strncmp(token, "pt2pt", strlen("pt2pt")))
            features |= OSHMPI_SET_MPI_GPU_FEATURE(OSHMPI_MPI_GPU_PT2PT);
        else if (!strncmp(token, "put", strlen("put")))
            features |= OSHMPI_SET_MPI_GPU_FEATURE(OSHMPI_MPI_GPU_PUT);
        else if (!strncmp(token, "get", strlen("get")))
            features |= OSHMPI_SET_MPI_GPU_FEATURE(OSHMPI_MPI_GPU_GET);
        else if (!strncmp(token, "acc", strlen("acc")))
            features |= OSHMPI_SET_MPI_GPU_FEATURE(OSHMPI_MPI_GPU_ACCUMULATES);
        token = (char *) strtok_r(NULL, ",", &savePtr);
    }

    /* update info only when any valid value is set */
    if (features)
        *features_ptr = features;
}

static void getstr_env_mpi_gpu_features(char *buf, size_t maxlen)
{
    int c = 0;

    OSHMPI_ASSERT(maxlen >= strlen("pt2pt,put,get,acc") + 1);

    if (OSHMPI_CHECK_MPI_GPU_FEATURE(OSHMPI_MPI_GPU_PT2PT))
        c += snprintf(buf + c, maxlen - c, "pt2pt");
    if (OSHMPI_CHECK_MPI_GPU_FEATURE(OSHMPI_MPI_GPU_PUT))
        c += snprintf(buf + c, maxlen - c, "%sput", (c > 0) ? "," : "");
    if (OSHMPI_CHECK_MPI_GPU_FEATURE(OSHMPI_MPI_GPU_GET))
        c += snprintf(buf + c, maxlen - c, "%sget", (c > 0) ? "," : "");
    if (OSHMPI_CHECK_MPI_GPU_FEATURE(OSHMPI_MPI_GPU_ACCUMULATES))
        c += snprintf(buf + c, maxlen - c, "%sacc", (c > 0) ? "," : "");

    if (c == 0)
        strncpy(buf, "none", maxlen);
}

#define STR_EXPAND(opts) #opts
#define STR(opts) STR_EXPAND(opts)
static void print_env(void)
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
        char mpi_gpu_features_str[256];
        OSHMPI_PRINTF("OSHMPI configuration:\n"
                      "    --enable-fast                "
#ifdef OSHMPI_FAST_OPTS
                      STR(OSHMPI_FAST_OPTS)","
#endif
#ifdef OSHMPI_DISABLE_DEBUG
                      "ndebug,"
#endif
#ifdef OSHMPI_ENABLE_IPO
                      "ipo,"
#endif
#if !defined(OSHMPI_FAST_OPTS) && !defined(OSHMPI_DISABLE_DEBUG) && !defined(OSHMPI_ENABLE_IPO)
                      "no"
#endif
                      "\n"
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
                      "    (initialized safety: %s)\n"
                      "    --enable-amo                 "
#ifdef OSHMPI_ENABLE_DIRECT_AMO
                      "direct\n"
#elif defined(OSHMPI_ENABLE_AM_AMO)
                      "am\n"
#else
                      "auto\n"
#endif
                      "    --enable-async-thread        "
#ifdef OSHMPI_ENABLE_AM_ASYNC_THREAD
                      "yes\n"
#elif defined(OSHMPI_DISABLE_AM_ASYNC_THREAD)
                      "no\n"
#else
                      "auto\n"
#endif
                "    --enable-rma        "
#ifdef OSHMPI_ENABLE_AM_RMA
                "am\n"
#elif defined(OSHMPI_ENABLE_DIRECT_RMA)
                "direct\n"
#else
                "auto\n"
#endif
                      "    --enable-op-tracking         "
#ifdef OSHMPI_ENABLE_OP_TRACKING
                      "yes\n"
#else
                      "no\n"
#endif
                      "    --enable-strided-cache         "
#ifdef OSHMPI_ENABLE_STRIDED_DTYPE_CACHE
                      "yes\n"
#else
                      "no\n"
#endif
                      "    --enable-win-type         "
#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
                      "dynamic_win\n"
#else
                      "win_creates\n"
#endif
                      "    --enable-cuda         "
#ifdef OSHMPI_ENABLE_CUDA
                      "yes\n"
#else
                      "no\n"
#endif
                      "    --enable-mpix-rma-abs         "
#ifdef OSHMPI_USE_MPIX_RMA_ABS
                      "yes\n"
#else
                      "no\n"
#endif
                      "\n",
                      OSHMPI_thread_level_str(OSHMPI_global.thread_level));

        getstr_env_amo_ops(OSHMPI_env.amo_ops, amo_ops_str, sizeof(amo_ops_str));
        getstr_env_mpi_gpu_features(mpi_gpu_features_str, sizeof(mpi_gpu_features_str));

        OSHMPI_PRINTF("OSHMPI environment variables:\n"
                      "    OSHMPI_VERBOSE               %d\n"
                      "    OSHMPI_AMO_OPS               %s\n"
                      "    OSHMPI_ENABLE_ASYNC_THREAD   %d\n"
                      "    OSHMPI_MPI_GPU_FEATURES      %s\n"
#ifndef OSHMPI_ENABLE_FAST
                      "    (Invalid options if OSHMPI is built with --enable-fast)\n"
                      "    OSHMPI_AMO_DBG_MODE          %s\n"
                      "    OSHMPI_RMA_DBG_MODE          %s\n"
#endif
                      ,OSHMPI_env.verbose, amo_ops_str,
                      OSHMPI_env.enable_async_thread,
                      mpi_gpu_features_str
#ifndef OSHMPI_ENABLE_FAST
                      ,getstr_env_dbg_mode(OSHMPI_env.amo_dbg_mode)
                      ,getstr_env_dbg_mode(OSHMPI_env.rma_dbg_mode)
#endif
);
    }
    /* *INDENT-ON* */
}

static void initialize_env(void)
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

    OSHMPI_env.mpi_gpu_features = 0;
    val = getenv("OSHMPI_MPI_GPU_FEATURES");
    if (val && strlen(val))
        set_env_mpi_gpu_features(val, &OSHMPI_env.mpi_gpu_features);
    else
        set_env_mpi_gpu_features("all", &OSHMPI_env.mpi_gpu_features);  /* default */

#ifndef OSHMPI_ENABLE_FAST
    OSHMPI_env.amo_dbg_mode = OSHMPI_DBG_AUTO;
    val = getenv("OSHMPI_AMO_DBG_MODE");
    if (val && strlen(val))
        set_env_dbg_mode(val, &OSHMPI_env.amo_dbg_mode);

    OSHMPI_env.rma_dbg_mode = OSHMPI_DBG_AUTO;
    val = getenv("OSHMPI_RMA_DBG_MODE");
    if (val && strlen(val))
        set_env_dbg_mode(val, &OSHMPI_env.rma_dbg_mode);
#endif

#ifdef OSHMPI_ENABLE_AM_ASYNC_THREAD
    OSHMPI_env.enable_async_thread = 1;
#elif defined(OSHMPI_DISABLE_AM_ASYNC_THREAD)
    OSHMPI_env.enable_async_thread = 0;
#else
    /* set default value based on AM status */
    if (!OSHMPI_ENABLE_DIRECT_AMO_RUNTIME || !OSHMPI_ENABLE_DIRECT_RMA_CONFIG)
        OSHMPI_env.enable_async_thread = 1;
    else
        OSHMPI_env.enable_async_thread = 0;
    /* check if overwritten by user setting */
    val = getenv("OSHMPI_ENABLE_ASYNC_THREAD");
    if (val && strlen(val))
        OSHMPI_env.enable_async_thread = atoi(val);
    if (OSHMPI_env.enable_async_thread != 0)
        OSHMPI_env.enable_async_thread = 1;
#endif
}

int OSHMPI_initialize_thread(int required, int *provided)
{
    int mpi_errno = MPI_SUCCESS;
    int mpi_provided = 0, mpi_initialized = 0, shm_provided = 0;

    if (OSHMPI_global.is_initialized)
        goto fn_exit;

    initialize_env();

    if (required != SHMEM_THREAD_SINGLE && required != SHMEM_THREAD_FUNNELED
        && required != SHMEM_THREAD_SERIALIZED && required != SHMEM_THREAD_MULTIPLE)
        OSHMPI_ERR_ABORT("Unknown OpenSHMEM thread support level: %d\n", required);

    /* Force thread multiple when async thread is enabled. */
    if (OSHMPI_ENABLE_AM_ASYNC_THREAD_RUNTIME)
        required = SHMEM_THREAD_MULTIPLE;

    shm_provided = required;

    /* Degrade thread safety if it is not set at configure */
#ifdef OSHMPI_ENABLE_THREAD_SINGLE
    if (required > SHMEM_THREAD_SINGLE)
        shm_provided = SHMEM_THREAD_SINGLE;
#elif defined(OSHMPI_ENABLE_THREAD_FUNNELED)
    if (required > SHMEM_THREAD_FUNNELED)
        shm_provided = SHMEM_THREAD_FUNNELED;
#elif defined(OSHMPI_ENABLE_THREAD_SERIALIZED)
    if (required > SHMEM_THREAD_SERIALIZED)
        shm_provided = SHMEM_THREAD_SERIALIZED;
#endif
    if (required != shm_provided)
        OSHMPI_DBGMSG("OSHMPI %s support is not enabled at configure (--enable-threads).\n",
                      OSHMPI_thread_level_str(required));

    OSHMPI_CALLMPI(MPI_Initialized(&mpi_initialized));
    if (mpi_initialized) {
        /* If MPI has already be initialized, we only query the thread safety. */
        OSHMPI_CALLMPI(MPI_Query_thread(&mpi_provided));
    } else {
        OSHMPI_CALLMPI(MPI_Init_thread(NULL, NULL, shm_provided, &mpi_provided));
    }

    /* Degrade thread safety if it is not supported by MPI */
    if (mpi_provided < shm_provided) {
        OSHMPI_DBGMSG("The MPI library does not support the required thread support:"
                      "required: %s, provided: %s.\n",
                      OSHMPI_thread_level_str(shm_provided), OSHMPI_thread_level_str(mpi_provided));
        shm_provided = mpi_provided;
    }

    if (OSHMPI_env.enable_async_thread && shm_provided < SHMEM_THREAD_MULTIPLE) {
        OSHMPI_ERR_ABORT("Asynchronous thread requires THREAD_MULTIPLE safety "
                         "but we cannot enable it at runtime. "
                         "Enable SHMEM_DEBUG to check the reason.\n");
    }

    /* OSHMPI internal routines are protected only when user explicitly requires multiple
     * safety, thus we do not expose the actual safety provided by MPI if it is higher. */
    OSHMPI_global.thread_level = shm_provided;

    /* Duplicate comm world for oshmpi use. */
    OSHMPI_CALLMPI(MPI_Comm_dup(MPI_COMM_WORLD, &OSHMPI_global.comm_world));
    OSHMPI_CALLMPI(MPI_Comm_size(OSHMPI_global.comm_world, &OSHMPI_global.world_size));
    OSHMPI_CALLMPI(MPI_Comm_rank(OSHMPI_global.comm_world, &OSHMPI_global.world_rank));
    OSHMPI_CALLMPI(MPI_Comm_group(OSHMPI_global.comm_world, &OSHMPI_global.comm_world_group));

    print_env();

    OSHMPI_global.page_sz = (size_t) sysconf(_SC_PAGESIZE);
    OSHMPIU_initialize_symm_mem(OSHMPI_global.comm_world);

#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
    initialize_symm_win();
    attach_symm_text();
    attach_symm_heap();
#else
    initialize_symm_text();
    initialize_symm_heap();
#endif

    OSHMPI_strided_initialize();
    OSHMPI_coll_initialize();
    OSHMPI_am_initialize();
    OSHMPI_space_initialize();

    OSHMPI_am_progress_mpi_barrier(OSHMPI_global.comm_world);
    OSHMPI_global.is_initialized = 1;

  fn_exit:
    if (provided)
        *provided = OSHMPI_global.thread_level;
    return mpi_errno;
}

static int finalize_impl(void)
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
    OSHMPI_am_finalize();
    OSHMPI_strided_finalize();
    OSHMPI_space_finalize();

#ifdef OSHMPI_ENABLE_DYNAMIC_WIN
    if (OSHMPI_global.symm_ictx.win != MPI_WIN_NULL) {
        OSHMPI_CALLMPI(MPI_Win_unlock_all(OSHMPI_global.symm_ictx.win));
        OSHMPI_CALLMPI(MPI_Win_detach(OSHMPI_global.symm_ictx.win,
                                      OSHMPI_global.symm_heap_attr.base));
        OSHMPI_CALLMPI(MPI_Win_detach(OSHMPI_global.symm_ictx.win,
                                      OSHMPI_global.symm_data_attr.base));
        OSHMPI_CALLMPI(MPI_Win_free(&OSHMPI_global.symm_ictx.win));
    }
    if (OSHMPI_SOBJ_HANDLE_CHECK_SYMMBIT(OSHMPI_global.symm_heap_attr.handle))
        OSHMPIU_free_symm_mem(OSHMPI_global.symm_heap_attr.base, OSHMPI_global.symm_heap_true_size);
    else
        OSHMPIU_free(OSHMPI_global.symm_heap_attr.base);
#else
    if (OSHMPI_global.symm_heap_ictx.win != MPI_WIN_NULL) {
        OSHMPI_CALLMPI(MPI_Win_unlock_all(OSHMPI_global.symm_heap_ictx.win));
        OSHMPI_CALLMPI(MPI_Win_free(&OSHMPI_global.symm_heap_ictx.win));
    }
    OSHMPI_THREAD_DESTROY_CS(&OSHMPI_global.symm_heap_mspace_cs);

    if (OSHMPI_global.symm_data_ictx.win != MPI_WIN_NULL) {
        OSHMPI_CALLMPI(MPI_Win_unlock_all(OSHMPI_global.symm_data_ictx.win));
        OSHMPI_CALLMPI(MPI_Win_free(&OSHMPI_global.symm_data_ictx.win));
    }
#endif /* end of OSHMPI_ENABLE_DYNAMIC_WIN */

    OSHMPI_sobj_destroy_attr(&OSHMPI_global.symm_data_attr);
    OSHMPI_sobj_destroy_attr(&OSHMPI_global.symm_heap_attr);
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

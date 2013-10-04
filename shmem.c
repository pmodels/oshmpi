/* -*- C -*-
 *
 * Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
 * 
 * This file is part of the Portals SHMEM software package. For license
 * information, see the LICENSE file in the top level directory of the
 * distribution.
 *
 */

/* This file was derived from shmem.h from the Portals4-SHMEM project.
 * All of the implementation code was written by Jeff Hammond.
 */

#include "shmem.h"

/* this code deals with SHMEM communication out of symmetric but non-heap data */
#if defined(_AIX)
    /* http://pic.dhe.ibm.com/infocenter/aix/v6r1/topic/com.ibm.aix.basetechref/doc/basetrf1/_end.htm */
    extern _end;
    extern _etext;
    extern _edata;
    unsigned long get_end()   { return _end;   }
    unsigned long get_etext() { return _etext; }
    unsigned long get_edata() { return _edata; }
#elif defined(__APPLE__)
    /* https://developer.apple.com/library/mac//documentation/Darwin/Reference/ManPages/10.7/man3/end.3.html */
#include <mach-o/getsect.h>
    unsigned long get_end();
    unsigned long get_etext();
    unsigned long get_edata();
#else
    /* http://man7.org/linux/man-pages/man3/end.3.html */
    extern etext;
    extern edata;
    extern end;
    unsigned long get_end()   { return end;   }
    unsigned long get_etext() { return etext; }
    unsigned long get_edata() { return edata; }
#endif

/*****************************************************************/
/* TODO convert all the global status into a struct ala ARMCI-MPI */
/* requires TLS if MPI is thread-based */
static MPI_Comm SHMEM_COMM_WORLD;
static MPI_Comm SHMEM_COMM_NODE;
static int      shmem_is_initialized = 0;
static int      shmem_is_finalized   = 0;
static int      shmem_world_is_smp   = 0;
static int      shmem_mpi_size, shmem_mpi_rank;

static MPI_Win sheap_win;
static int     sheap_is_symmetric;
static int     sheap_size;
static void *  sheap_mybase_ptr;
static void ** sheap_base_ptrs;
/*****************************************************************/

/* 8.1: Initialization Routines */
static void __shmem_initialize(void)
{
    int flag, provided;
    MPI_Initialized(&flag);
    if (!flag) 
        MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &provided);

    if (!shmem_is_initialized) {

        MPI_Comm_dup(MPI_COMM_WORLD, &SHMEM_COMM_WORLD);
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0 /* key */, MPI_INFO_NULL, &SHMEM_COMM_NODE);

        int result;
        MPI_Comm_compare(SHMEM_COMM_WORLD, SHMEM_COMM_NODE, &result);
        shmem_world_is_smp = (result==MPI_IDENT || result==MPI_CONGRUENT) ? 1 : 0;

        MPI_Comm_size(SHMEM_COMM_WORLD, &shmem_mpi_size);
        MPI_Comm_rank(SHMEM_COMM_WORLD, &shmem_mpi_rank);

        char * c = getenv("SHMEM_SYMMETRIC_HEAP_SIZE");
        sheap_size = ( (c) ? atoi(c) : 128*1024*1024 );
        MPI_Info info = MPI_INFO_NULL;
        void * mybase = NULL;

        /* TODO something for shared memory windows when comm_world == comm_shared */

        MPI_Win_allocate((MPI_Aint)sheap_size, 1 /* disp_unit */, info, SHMEM_COMM_WORLD, &mybase, &sheap_win);

        /* I am not sure if there is a better way to operate on addresses... */
        void * minbase;
        void * maxbase;
        /* cannot fuse allreduces because max{base,-base} trick does not work for unsigned */
        MPI_Allreduce( &mybase, &minbase, 1, sizeof(void*)==4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG, MPI_MIN, SHMEM_COMM_WORLD );
        MPI_Allreduce( &mybase, &maxbase, 1, sizeof(void*)==4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG, MPI_MAX, SHMEM_COMM_WORLD );
        sheap_is_symmetric = (minbase==mybase && mybase==maxbase) ? 1 : 0;

        if (!sheap_is_symmetric) {
            /* non-symmetric heap requires O(nproc) metadata */
            MPI_Alloc_mem(shmem_mpi_size*sizeof(void*), MPI_INFO_NULL, &sheap_base_ptrs);
            MPI_Allgather(&mybase, sizeof(void*), MPI_BYTE, sheap_base_ptrs, sizeof(void*), MPI_BYTE, SHMEM_COMM_WORLD);
        } else {
            sheap_mybase_ptr = mybase;
        }

        /* TODO deal with non-sheap memory (MPI_Win_create) */
        get_end();
        get_etext();
        get_edata();

        shmem_is_initialized = 1;
    }

    return;
}

static void __shmem_abort(int code)
{
    MPI_Abort(SHMEM_COMM_WORLD, code);
    return;
}

static void __shmem_finalize(void)
{
    int flag;
    MPI_Finalized(&flag);

    if (!flag) {
        if (shmem_is_initialized && !shmem_is_finalized) {
            if (!sheap_is_symmetric) 
                MPI_Free_mem(sheap_base_ptrs);

            MPI_Win_free(&sheap_win);
            MPI_Comm_free(&SHMEM_COMM_NODE);
            MPI_Comm_free(&SHMEM_COMM_WORLD);

            shmem_is_finalized = 1;
        }

        MPI_Finalize();
    }

    return;
}

void start_pes(int npes) { 
    __shmem_initialize(); 
    atexit(__shmem_finalize);
    return;
}

/* 8.2: Query Routines */
int _num_pes(void) { return shmem_mpi_size; }
int shmem_n_pes(void) { return shmem_mpi_size; }
int _my_pe(void) { return shmem_mpi_rank; }
int shmem_my_pe(void) { return shmem_mpi_rank; }

/* 8.3: Accessibility Query Routines */
int shmem_pe_accessible(int pe) { return ( 0<=pe && pe<=shmem_mpi_size ); } 
int shmem_addr_accessible(void *addr, int pe) 
{ 
    if (shmem_pe_accessible(pe)) {
        /* TODO check address accessibility */
        return 1;
    } else {
        return 0;
    }
}

/* 8.4: Symmetric Heap Routines */
static inline void __shmem_window_offset(void *target, int pe,                  /* IN  */
                                         int * window, shmem_offset_t * offset) /* OUT */
{
    /* it would be nice if this code avoided evil casting... */
    if (0 /* test for text/data */) {
    } else /* symmetric heap */ {
        if (sheap_is_symmetric) {
            ptrdiff_t offset = target - sheap_mybase_ptr;
        } else {
            ptrdiff_t offset = target - sheap_base_ptrs[pe];    
        }
        assert((uint64_t)offset<(uint64_t)INT32_MAX); /* supporting offset bigger than max int requires more code */
    }
    return;
}

void *shmalloc(size_t size);
void *shmemalign(size_t alignment, size_t size);
void *shrealloc(void *ptr, size_t size);
void shfree(void *ptr);

/* 8.5: Remote Pointer Operations */
void *shmem_ptr(void *target, int pe)
{ 
    /* TODO shared memory window optimization */
    return NULL; 
}

static inline void __shmem_put(void *target, const void *source, size_t len, int pe)
{
    //shmem_offset_t offset = __shmem_symmetric_heap_offset(void *target, int pe);

    return;
}


/* 8.6: Elemental Put Routines */
void shmem_float_p(float *addr, float value, int pe);
void shmem_double_p(double *addr, double value, int pe);
void shmem_longdouble_p(long double *addr, long double value, int pe);
void shmem_char_p(char *addr, char value, int pe);
void shmem_short_p(short *addr, short value, int pe);
void shmem_int_p(int *addr, int value, int pe);
void shmem_long_p(long *addr, long value, int pe);
void shmem_longlong_p(long long *addr, long long value, int pe);

/* 8.7: Block Data Put Routines */
void shmem_putmem(void *target, const void *source, size_t len, int pe);
void shmem_float_put(float *target, const float *source, size_t len, int pe);
void shmem_double_put(double *target, const double *source, size_t len, int pe);
void shmem_longdouble_put(long double *target, const long double *source, size_t len, int pe);
void shmem_char_put(char *target, const char *source, size_t nelems, int pe);
void shmem_short_put(short *target, const short *source, size_t len, int pe);
void shmem_int_put(int *target, const int *source, size_t len, int pe);
void shmem_long_put(long *target, const long *source, size_t len, int pe);
void shmem_longlong_put(long long *target, const long long *source, size_t len, int pe);
void shmem_put32(void *target, const void *source, size_t len, int pe);
void shmem_put64(void *target, const void *source, size_t len, int pe);
void shmem_put128(void *target, const void *source, size_t len, int pe);
void shmem_complexf_put(float complex * target, const float complex * source, size_t nelems, int pe);
void shmem_complexd_put(double complex * target, const double complex * source, size_t nelems, int pe);

/* 8.8: Strided Put Routines */
void shmem_float_iput(float *target, const float *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_double_iput(double *target, const double *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_longdouble_iput(long double *target, const long double *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_short_iput(short *target, const short *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_int_iput(int *target, const int *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_long_iput(long *target, const long *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_longlong_iput(long long *target, const long long *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_iput32(void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_iput64(void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_iput128(void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);

/* 8.9: Elemental Data Get Routines */
float shmem_float_g(float *addr, int pe);
double shmem_double_g(double *addr, int pe);
long double shmem_longdouble_g(long double *addr, int pe);
char shmem_char_g(char *addr, int pe);
short shmem_short_g(short *addr, int pe);
int shmem_int_g(int *addr, int pe);
long shmem_long_g(long *addr, int pe);
long long shmem_longlong_g(long long *addr, int pe);

/* 8.10 Block Data Get Routines */
void shmem_getmem(void *target, const void *source, size_t len, int pe);
void shmem_float_get(float *target, const float *source, size_t len, int pe);
void shmem_double_get(double *target, const double *source, size_t len, int pe);
void shmem_longdouble_get(long double *target, const long double *source, size_t len, int pe);
void shmem_char_get(char *target, const char *source, size_t len, int pe);
void shmem_short_get(short *target, const short *source, size_t len, int pe);
void shmem_int_get(int *target, const int *source, size_t len, int pe);
void shmem_long_get(long *target, const long *source, size_t len, int pe);
void shmem_longlong_get(long long *target, const long long *source, size_t len, int pe);
void shmem_get32(void *target, const void *source, size_t len, int pe);
void shmem_get64(void *target, const void *source, size_t len, int pe);
void shmem_get128(void *target, const void *source, size_t len, int pe);
void shmem_complexf_get(float complex * target, const float complex * source, size_t nelems, int pe);
void shmem_complexd_get(double complex * target, const double complex * source, size_t nelems, int pe);

/* 8.11: Strided Get Routines */
void shmem_float_iget(float *target, const float *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_double_iget(double *target, const double *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_longdouble_iget(long double *target, const long double *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_short_iget(short *target, const short *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_int_iget(int *target, const int *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_long_iget(long *target, const long *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_longlong_iget(long long *target, const long long *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_iget32(void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_iget64(void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
void shmem_iget128(void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);

/* 8.12: Atomic Memory fetch-and-operate Routines -- Swap */
float shmem_float_swap(float *target, float value, int pe);
double shmem_double_swap(double *target, double value, int pe);
int shmem_int_swap(int *target, int value, int pe);
long shmem_long_swap(long *target, long value, int pe);
long long shmem_longlong_swap(long long *target, long long value, int pe);
long shmem_swap(long *target, long value, int pe);

/* 8.12: Atomic Memory fetch-and-operate Routines -- Cswap */
int shmem_int_cswap(int *target, int cond, int value, int pe);
long shmem_long_cswap(long *target, long cond, long value, int pe);
long long shmem_longlong_cswap(long long * target, long long cond, 
                          long long value, int pe);

/* 8.12: Atomic Memory fetch-and-operate Routines -- Fetch and Add */
int shmem_int_fadd(int *target, int value, int pe);
long shmem_long_fadd(long *target, long value, int pe);
long long shmem_longlong_fadd(long long *target, long long value, int pe);

/* 8.12: Atomic Memory fetch-and-operate Routines -- Fetch and Increment */
int shmem_int_finc(int *target, int pe);
long shmem_long_finc(long *target, int pe);
long long shmem_longlong_finc(long long *target, int pe);

/* 8.13: Atomic Memory Operation Routines -- Add */
void shmem_int_add(int *target, int value, int pe);
void shmem_long_add(long *target, long value, int pe);
void shmem_longlong_add(long long *target, long long value, int pe);

/* 8.13: Atomic Memory Operation Routines -- Increment */
void shmem_int_inc(int *target, int pe);
void shmem_long_inc(long *target, int pe);
void shmem_longlong_inc(long long *target, int pe);

/* 8.14: Point-to-Point Synchronization Routines -- Wait*/
void shmem_short_wait(short *var, short value);
void shmem_int_wait(int *var, int value);
void shmem_long_wait(long *var, long value);
void shmem_longlong_wait(long long *var, long long value);
void shmem_wait(long *ivar, long cmp_value);

/* 8.14: Point-to-Point Synchronization Routines -- Wait Until*/
void shmem_short_wait_until(short *var, int cond, short value);
void shmem_int_wait_until(int *var, int cond, int value);
void shmem_long_wait_until(long *var, int cond, long value);
void shmem_longlong_wait_until(long long *var, int cond, long long value);
void shmem_wait_until(long *ivar, int cmp, long value);

/* 8.15: Barrier Synchronization Routines */
void shmem_barrier(int PE_start, int logPE_stride, int PE_size, long *pSync);
void shmem_barrier_all(void);

void shmem_quiet(void);
void shmem_fence(void);

/* 8.16: Reduction Routines */
void shmem_short_and_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync);
void shmem_int_and_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync);
void shmem_long_and_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync);
void shmem_longlong_and_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync);

void shmem_short_or_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync);
void shmem_int_or_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync);
void shmem_long_or_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync);
void shmem_longlong_or_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync);

void shmem_short_xor_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync);
void shmem_int_xor_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync);
void shmem_long_xor_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync);
void shmem_longlong_xor_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync);

void shmem_float_min_to_all(float *target, float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync);
void shmem_double_min_to_all(double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync);
void shmem_longdouble_min_to_all(long double *target, long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync);
void shmem_short_min_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync);
void shmem_int_min_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync);
void shmem_long_min_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync);
void shmem_longlong_min_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync);

void shmem_float_max_to_all(float *target, float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync);
void shmem_double_max_to_all(double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync);
void shmem_longdouble_max_to_all(long double *target, long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync);
void shmem_short_max_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync);
void shmem_int_max_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync);
void shmem_long_max_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync);
void shmem_longlong_max_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync);

void shmem_float_sum_to_all(float *target, float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync);
void shmem_double_sum_to_all(double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync);
void shmem_longdouble_sum_to_all(long double *target, long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync);
void shmem_complexf_sum_to_all(float complex *target, float complex *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float complex *pWrk, long *pSync);
void shmem_complexd_sum_to_all(double complex *target, double complex *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double complex *pWrk, long *pSync);
void shmem_short_sum_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync);
void shmem_int_sum_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync);
void shmem_long_sum_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync);
void shmem_longlong_sum_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync);

void shmem_float_prod_to_all(float *target, float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync);
void shmem_double_prod_to_all(double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync);
void shmem_longdouble_prod_to_all(long double *target, long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync);
void shmem_complexf_prod_to_all(float complex *target, float complex *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float complex *pWrk, long *pSync);
void shmem_complexd_prod_to_all(double complex *target, double complex *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double complex *pWrk, long *pSync);
void shmem_short_prod_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync);
void shmem_int_prod_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync);
void shmem_long_prod_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync);
void shmem_longlong_prod_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync);

/* 8.17: Collect Routines */
void shmem_collect32(void *target, const void *source, size_t nlong, int PE_start, int logPE_stride, int PE_size, long *pSync);
void shmem_collect64(void *target, const void *source, size_t nlong, int PE_start, int logPE_stride, int PE_size, long *pSync);
void shmem_fcollect32(void *target, const void *source, size_t nlong, int PE_start, int logPE_stride, int PE_size, long *pSync);
void shmem_fcollect64(void *target, const void *source, size_t nlong, int PE_start, int logPE_stride, int PE_size, long *pSync);

/* 8.18: Broadcast Routines */
void shmem_broadcast32(void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync);
void shmem_broadcast64(void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync);

/* 8.19: Lock Routines */
void shmem_set_lock(long *lock);
void shmem_clear_lock(long *lock);
int shmem_test_lock(long *lock);

/* A.1: Cache Management Routines (deprecated) */
void shmem_set_cache_inv(void) { return; }
void shmem_set_cache_line_inv(void *target) { return; }
void shmem_clear_cache_inv(void) { return; }
void shmem_clear_cache_line_inv(void *target) { return; }
void shmem_udcflush(void) { return; }
void shmem_udcflush_line(void *target) { return; }


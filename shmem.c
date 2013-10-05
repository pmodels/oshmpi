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

#if 0 //( defined(__GNUC__) && (__GNUC__ >= 3) ) || defined(__IBMC__) || defined(__INTEL_COMPILER) || defined(__clang__)
#  define unlikely(x_) __builtin_expect(!!(x_),0)
#  define likely(x_)   __builtin_expect(!!(x_),1)
#else
#  define unlikely(x_) (x_)
#  define likely(x_)   (x_)
#endif

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
static MPI_Comm  SHMEM_COMM_WORLD;
static MPI_Comm  SHMEM_COMM_NODE;
static MPI_Group SHMEM_GROUP_WORLD; /* used for creating logpe comms */
static int       shmem_is_initialized = 0;
static int       shmem_is_finalized   = 0;
static int       shmem_world_is_smp   = 0;
static int       shmem_mpi_size, shmem_mpi_rank;
static char      shmem_procname[MPI_MAX_PROCESSOR_NAME];

/* TODO probably want to make these 5 things into a struct typedef */
static MPI_Win shmem_etext_win;
static int     shmem_etext_is_symmetric;
static int     shmem_etext_size;
static void *  shmem_etext_mybase_ptr;
static void ** shmem_etext_base_ptrs;

static MPI_Win shmem_sheap_win;
static int     shmem_sheap_is_symmetric;
static int     shmem_sheap_size;
static void *  shmem_sheap_mybase_ptr;
static void ** shmem_sheap_base_ptrs;
/*****************************************************************/

enum shmem_window_id_e { SHMEM_SHEAP_WINDOW = 0, SHMEM_ETEXT_WINDOW = 1 };
enum shmem_rma_type_e  { SHMEM_PUT = 0, SHMEM_GET = 1, SHMEM_IPUT = 2, SHMEM_IGET = 4};
enum shmem_amo_type_e  { SHMEM_SWAP = 0, SHMEM_CSWAP = 1, SHMEM_ADD = 2, SHMEM_FADD = 4};
enum shmem_coll_type_e { SHMEM_BARRIER = 0, SHMEM_BROADCAST = 1, SHMEM_ALLREDUCE = 2, SHMEM_ALLGATHER = 4, SHMEM_ALLGATHERV = 5};

/* 8.1: Initialization Routines */
static int __shmem_address_is_symmetric(void * my_sheap_base_ptr)
{
    /* I am not sure if there is a better way to operate on addresses... */

    void * minbase;
    void * maxbase;

    /* cannot fuse allreduces because max{base,-base} trick does not work for unsigned */
    MPI_Allreduce( &my_sheap_base_ptr, &minbase, 1, sizeof(void*)==4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG, MPI_MIN, SHMEM_COMM_WORLD );
    MPI_Allreduce( &my_sheap_base_ptr, &maxbase, 1, sizeof(void*)==4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG, MPI_MAX, SHMEM_COMM_WORLD );

    return ((minbase==my_sheap_base_ptr && my_sheap_base_ptr==maxbase) ? 1 : 0);
}

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

        MPI_Comm_group(SHMEM_COMM_WORLD, &SHMEM_GROUP_WORLD);

        char * c = getenv("SHMEM_SYMMETRIC_HEAP_SIZE");
        shmem_sheap_size = ( (c) ? atoi(c) : 128*1024*1024 );
        MPI_Info info = MPI_INFO_NULL; /* TODO set info keys to disable unnecessary accumulate support */
        void * my_sheap_base_ptr = NULL;

        /* TODO something for shared memory windows when comm_world == comm_shared */

        MPI_Win_allocate((MPI_Aint)shmem_sheap_size, 1 /* disp_unit */, info, SHMEM_COMM_WORLD, &my_sheap_base_ptr, &shmem_sheap_win);

        shmem_sheap_is_symmetric = __shmem_address_is_symmetric(my_sheap_base_ptr);

        if (!shmem_sheap_is_symmetric) {
            /* non-symmetric heap requires O(nproc) metadata */
            shmem_sheap_base_ptrs = malloc(shmem_mpi_size*sizeof(void*)); assert(shmem_sheap_base_ptrs!=NULL);
            MPI_Allgather(&my_sheap_base_ptr, sizeof(void*), MPI_BYTE, shmem_sheap_base_ptrs, sizeof(void*), MPI_BYTE, SHMEM_COMM_WORLD);
        } else {
            shmem_sheap_mybase_ptr = my_sheap_base_ptr;
        }

        /* TODO deal with non-sheap memory (MPI_Win_create) */
        get_end();
        get_etext();
        get_edata();

        void * my_etext_base_ptr = NULL;

        shmem_etext_is_symmetric = __shmem_address_is_symmetric(my_etext_base_ptr);

        if (!shmem_etext_is_symmetric) {
            /* non-symmetric heap requires O(nproc) metadata */
            shmem_etext_base_ptrs = malloc(shmem_mpi_size*sizeof(void*)); assert(shmem_etext_base_ptrs!=NULL);
            MPI_Allgather(&my_sheap_base_ptr, sizeof(void*), MPI_BYTE, shmem_etext_base_ptrs, sizeof(void*), MPI_BYTE, SHMEM_COMM_WORLD);
        } else {
            shmem_etext_mybase_ptr = my_sheap_base_ptr;
        }

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
            if (!shmem_sheap_is_symmetric) 
                MPI_Free_mem(shmem_sheap_base_ptrs);
            if (!shmem_etext_is_symmetric) 
                MPI_Free_mem(shmem_etext_base_ptrs);

            MPI_Win_free(&shmem_sheap_win);

            MPI_Group_free(&SHMEM_GROUP_WORLD);

            MPI_Comm_free(&SHMEM_COMM_NODE);
            MPI_Comm_free(&SHMEM_COMM_WORLD);

            shmem_is_finalized = 1;
        }

        MPI_Finalize();
    }

    return;
}

void start_pes(int npes) 
{ 
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
static inline void __shmem_window_offset(const void *target, const int pe, /* IN  */
                                         enum shmem_window_id_e * win_id,  /* OUT */
                                         shmem_offset_t * offset)          /* OUT */
{
    /* it would be nice if this code avoided evil casting... */
    if (0 /* test for text/data */) {
        /* TODO */
        *win_id = SHMEM_ETEXT_WINDOW;
    } else /* symmetric heap */ {
        if (shmem_sheap_is_symmetric) {
            *offset = target - shmem_sheap_mybase_ptr;
        } else {
            *offset = target - shmem_sheap_base_ptrs[pe];    
        }
        assert((uint64_t)(*offset)<(uint64_t)INT32_MAX); /* supporting offset bigger than max int requires more code */

        *win_id = SHMEM_SHEAP_WINDOW;
    }
    return;
}

void *shmalloc(size_t size);
void *shmemalign(size_t alignment, size_t size);
void *shrealloc(void *ptr, size_t size);
void shfree(void *ptr);

void shmem_quiet(void);
void shmem_fence(void);

/* 8.5: Remote Pointer Operations */
void *shmem_ptr(void *target, int pe)
{ 
    /* TODO shared memory window optimization */
    return NULL; 
}

static inline void __shmem_rma(enum shmem_rma_type_e rma, MPI_Datatype mpi_type,
                               void *target, const void *source, size_t len, int pe)
{
    enum shmem_window_id_e win_id;
    shmem_offset_t win_offset;

    int count = 0;
    if ( likely(len<(size_t)INT32_MAX) ) {
        count = len;
    } else {
        /* TODO generate derived type ala BigMPI */
        __shmem_abort(rma);
    }

    switch (rma) {
        case SHMEM_PUT:
            __shmem_window_offset(target, pe, &win_id, &win_offset);
            MPI_Put(source, count, mpi_type, pe, (MPI_Aint)win_offset, count, mpi_type,
                    (win_id==SHMEM_ETEXT_WINDOW) ? shmem_etext_win : shmem_sheap_win);
            break;
        case SHMEM_GET:
            __shmem_window_offset(source, pe, &win_id, &win_offset);
            MPI_Get(target, count, mpi_type, pe, (MPI_Aint)win_offset, count, mpi_type,
                    (win_id==SHMEM_ETEXT_WINDOW) ? shmem_etext_win : shmem_sheap_win);
            break;
#if 0
        case SHMEM_IPUT:
            __shmem_window_offset(target, pe, &win_id, &win_offset);
            MPI_Put(source, count, mpi_type, pe, (MPI_Aint)win_offset, count, mpi_type,
                    (win_id==SHMEM_ETEXT_WINDOW) ? shmem_etext_win : shmem_sheap_win);
            break;
        case SHMEM_IGET:
            __shmem_window_offset(source, pe, &win_id, &win_offset);
            MPI_Get(source, count, mpi_type, pe, (MPI_Aint)win_offset, count, mpi_type,
                    (win_id==SHMEM_ETEXT_WINDOW) ? shmem_etext_win : shmem_sheap_win);
            break;
#endif
        default:
            __shmem_abort(rma);
            break;
    }
    return;
}

/* 8.6: Elemental Put Routines */
void shmem_float_p(float *addr, float v, int pe)                  { __shmem_rma(SHMEM_PUT, MPI_FLOAT,       addr, &v, 1, pe); }
void shmem_double_p(double *addr, double v, int pe)               { __shmem_rma(SHMEM_PUT, MPI_DOUBLE,      addr, &v, 1, pe); }
void shmem_longdouble_p(long double *addr, long double v, int pe) { __shmem_rma(SHMEM_PUT, MPI_LONG_DOUBLE, addr, &v, 1, pe); }
void shmem_char_p(char *addr, char v, int pe)                     { __shmem_rma(SHMEM_PUT, MPI_CHAR,        addr, &v, 1, pe); }
void shmem_short_p(short *addr, short v, int pe)                  { __shmem_rma(SHMEM_PUT, MPI_SHORT,       addr, &v, 1, pe); }
void shmem_int_p(int *addr, int v, int pe)                        { __shmem_rma(SHMEM_PUT, MPI_INT,         addr, &v, 1, pe); }
void shmem_long_p(long *addr, long v, int pe)                     { __shmem_rma(SHMEM_PUT, MPI_LONG,        addr, &v, 1, pe); }
void shmem_longlong_p(long long *addr, long long v, int pe)       { __shmem_rma(SHMEM_PUT, MPI_LONG_LONG,   addr, &v, 1, pe); }

/* 8.7: Block Data Put Routines */
void shmem_float_put(float *target, const float *source, size_t len, int pe)                           { __shmem_rma(SHMEM_PUT, MPI_FLOAT,          target, source, len, pe); }
void shmem_double_put(double *target, const double *source, size_t len, int pe)                        { __shmem_rma(SHMEM_PUT, MPI_DOUBLE,         target, source, len, pe); }
void shmem_longdouble_put(long double *target, const long double *source, size_t len, int pe)          { __shmem_rma(SHMEM_PUT, MPI_LONG_DOUBLE,    target, source, len, pe); }
void shmem_char_put(char *target, const char *source, size_t len, int pe)                              { __shmem_rma(SHMEM_PUT, MPI_CHAR,           target, source, len, pe); }
void shmem_short_put(short *target, const short *source, size_t len, int pe)                           { __shmem_rma(SHMEM_PUT, MPI_SHORT,          target, source, len, pe); }
void shmem_int_put(int *target, const int *source, size_t len, int pe)                                 { __shmem_rma(SHMEM_PUT, MPI_INT,            target, source, len, pe); }
void shmem_long_put(long *target, const long *source, size_t len, int pe)                              { __shmem_rma(SHMEM_PUT, MPI_LONG,           target, source, len, pe); }    
void shmem_longlong_put(long long *target, const long long *source, size_t len, int pe)                { __shmem_rma(SHMEM_PUT, MPI_LONG_LONG,      target, source, len, pe); }
void shmem_putmem(void *target, const void *source, size_t len, int pe)                                { __shmem_rma(SHMEM_PUT, MPI_BYTE,           target, source, len, pe); }
void shmem_put32(void *target, const void *source, size_t len, int pe)                                 { __shmem_rma(SHMEM_PUT, MPI_INT32_T,        target, source, len, pe); }
void shmem_put64(void *target, const void *source, size_t len, int pe)                                 { __shmem_rma(SHMEM_PUT, MPI_INT64_T,        target, source, len, pe); }
void shmem_put128(void *target, const void *source, size_t len, int pe)                                { __shmem_rma(SHMEM_PUT, MPI_INT64_T,        target, source, 2*len, pe); }
void shmem_complexf_put(float complex * target, const float complex * source, size_t len, int pe)      { __shmem_rma(SHMEM_PUT, MPI_COMPLEX,        target, source, len, pe); }
void shmem_complexd_put(double complex * target, const double complex * source, size_t len, int pe)    { __shmem_rma(SHMEM_PUT, MPI_DOUBLE_COMPLEX, target, source, len, pe); }

/* 8.9: Elemental Data Get Routines */
float       shmem_float_g(float *addr, int pe)            { float       v; __shmem_rma(SHMEM_GET, MPI_FLOAT,       addr, &v, 1, pe); return v; }
double      shmem_double_g(double *addr, int pe)          { double      v; __shmem_rma(SHMEM_GET, MPI_DOUBLE,      addr, &v, 1, pe); return v; }
long double shmem_longdouble_g(long double *addr, int pe) { long double v; __shmem_rma(SHMEM_GET, MPI_LONG_DOUBLE, addr, &v, 1, pe); return v; }
char        shmem_char_g(char *addr, int pe)              { char        v; __shmem_rma(SHMEM_GET, MPI_CHAR,        addr, &v, 1, pe); return v; }
short       shmem_short_g(short *addr, int pe)            { short       v; __shmem_rma(SHMEM_GET, MPI_SHORT,       addr, &v, 1, pe); return v; }
int         shmem_int_g(int *addr, int pe)                { int         v; __shmem_rma(SHMEM_GET, MPI_INT,         addr, &v, 1, pe); return v; }
long        shmem_long_g(long *addr, int pe)              { long        v; __shmem_rma(SHMEM_GET, MPI_LONG,        addr, &v, 1, pe); return v; }
long long   shmem_longlong_g(long long *addr, int pe)     { long long   v; __shmem_rma(SHMEM_GET, MPI_LONG_LONG,   addr, &v, 1, pe); return v; }

/* 8.10 Block Data Get Routines */
void shmem_float_get(float *target, const float *source, size_t len, int pe)                           { __shmem_rma(SHMEM_GET, MPI_FLOAT,          target, source, len, pe); }
void shmem_double_get(double *target, const double *source, size_t len, int pe)                        { __shmem_rma(SHMEM_GET, MPI_DOUBLE,         target, source, len, pe); }
void shmem_longdouble_get(long double *target, const long double *source, size_t len, int pe)          { __shmem_rma(SHMEM_GET, MPI_LONG_DOUBLE,    target, source, len, pe); }
void shmem_char_get(char *target, const char *source, size_t len, int pe)                              { __shmem_rma(SHMEM_GET, MPI_CHAR,           target, source, len, pe); }
void shmem_short_get(short *target, const short *source, size_t len, int pe)                           { __shmem_rma(SHMEM_GET, MPI_SHORT,          target, source, len, pe); }
void shmem_int_get(int *target, const int *source, size_t len, int pe)                                 { __shmem_rma(SHMEM_GET, MPI_INT,            target, source, len, pe); }
void shmem_long_get(long *target, const long *source, size_t len, int pe)                              { __shmem_rma(SHMEM_GET, MPI_LONG,           target, source, len, pe); }    
void shmem_longlong_get(long long *target, const long long *source, size_t len, int pe)                { __shmem_rma(SHMEM_GET, MPI_LONG_LONG,      target, source, len, pe); }
void shmem_getmem(void *target, const void *source, size_t len, int pe)                                { __shmem_rma(SHMEM_GET, MPI_BYTE,           target, source, len, pe); }
void shmem_get32(void *target, const void *source, size_t len, int pe)                                 { __shmem_rma(SHMEM_GET, MPI_INT32_T,        target, source, len, pe); }
void shmem_get64(void *target, const void *source, size_t len, int pe)                                 { __shmem_rma(SHMEM_GET, MPI_INT64_T,        target, source, len, pe); }
void shmem_get128(void *target, const void *source, size_t len, int pe)                                { __shmem_rma(SHMEM_GET, MPI_INT64_T,        target, source, 2*len, pe); }
void shmem_complexf_get(float complex * target, const float complex * source, size_t len, int pe)      { __shmem_rma(SHMEM_GET, MPI_COMPLEX,        target, source, len, pe); }
void shmem_complexd_get(double complex * target, const double complex * source, size_t len, int pe)    { __shmem_rma(SHMEM_GET, MPI_DOUBLE_COMPLEX, target, source, len, pe); }

#if 0
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
#endif

/* TODO
 * AMO implementations would benefit greatly from specialization and/or the use of macros to
 * (1) allow for a return value rather than stack temporary, and
 * (2) eliminate the stack temporary in the case of (F)INC using (F)ADD. */

static inline void __shmem_amo(enum shmem_amo_type_e amo, MPI_Datatype mpi_type,
                               void *output,        /* not used for ADD */
                               void *remote, 
                               const void *input,
                               const void *compare, /* only used for CSWAP */
                               int pe)
{
    enum shmem_window_id_e win_id;
    shmem_offset_t win_offset;

    switch (amo) {
        case SHMEM_SWAP:
            __shmem_window_offset(remote, pe, &win_id, &win_offset);
            MPI_Fetch_and_op(input, output, mpi_type, pe, win_offset, MPI_REPLACE, 
                             (win_id==SHMEM_ETEXT_WINDOW) ? shmem_etext_win : shmem_sheap_win);
            break;
        case SHMEM_CSWAP:
            __shmem_window_offset(remote, pe, &win_id, &win_offset);
            MPI_Compare_and_swap(input, compare, output, mpi_type, pe, win_offset,
                                 (win_id==SHMEM_ETEXT_WINDOW) ? shmem_etext_win : shmem_sheap_win);
            break;
        /* (F)INC = (F)ADD w/ input=1 at the higher level */
        case SHMEM_ADD:
            __shmem_window_offset(remote, pe, &win_id, &win_offset);
            MPI_Fetch_and_op(input, NULL, mpi_type, pe, win_offset, MPI_SUM, 
                             (win_id==SHMEM_ETEXT_WINDOW) ? shmem_etext_win : shmem_sheap_win);
            break;
        case SHMEM_FADD:
            __shmem_window_offset(remote, pe, &win_id, &win_offset);
            MPI_Fetch_and_op(input, output, mpi_type, pe, win_offset, MPI_SUM, 
                             (win_id==SHMEM_ETEXT_WINDOW) ? shmem_etext_win : shmem_sheap_win);
            break;
        default:
            __shmem_abort(amo);
            break;
    }
    return;
}

/* Naming conventions for shorthand:
 * r = return v
 * c = comparand
 * v = v (input)
 * t = target 
 */

/* 8.12: Atomic Memory fetch-and-operate Routines -- Swap */
float     shmem_float_swap(float *t, float v, int pe)            { float     r; __shmem_amo(SHMEM_SWAP, MPI_FLOAT,     &r, t, &v, NULL, pe) ; return r; }
double    shmem_double_swap(double *t, double v, int pe)         { double    r; __shmem_amo(SHMEM_SWAP, MPI_DOUBLE,    &r, t, &v, NULL, pe) ; return r; }
int       shmem_int_swap(int *t, int v, int pe)                  { int       r; __shmem_amo(SHMEM_SWAP, MPI_INT,       &r, t, &v, NULL, pe) ; return r; }
long      shmem_long_swap(long *t, long v, int pe)               { long      r; __shmem_amo(SHMEM_SWAP, MPI_LONG,      &r, t, &v, NULL, pe) ; return r; }
long long shmem_longlong_swap(long long *t, long long v, int pe) { long long r; __shmem_amo(SHMEM_SWAP, MPI_LONG_LONG, &r, t, &v, NULL, pe) ; return r; }
long      shmem_swap(long *t, long v, int pe)                    { long      r; __shmem_amo(SHMEM_SWAP, MPI_LONG,      &r, t, &v, NULL, pe) ; return r; }

/* 8.12: Atomic Memory fetch-and-operate Routines -- Cswap */
int       shmem_int_cswap(int *t, int c, int v, int pe)                         { int       r; __shmem_amo(SHMEM_CSWAP, MPI_INT,       &r, t, &v, &c, pe) ; return r; }
long      shmem_long_cswap(long *t, long c, long v, int pe)                     { long      r; __shmem_amo(SHMEM_CSWAP, MPI_LONG,      &r, t, &v, &c, pe) ; return r; }
long long shmem_longlong_cswap(long long * t, long long c, long long v, int pe) { long long r; __shmem_amo(SHMEM_CSWAP, MPI_LONG_LONG, &r, t, &v, &c, pe) ; return r; }

/* 8.12: Atomic Memory fetch-and-operate Routines -- Fetch and Add */
int       shmem_int_fadd(int *t, int v, int pe)                  { int       r; __shmem_amo(SHMEM_FADD, MPI_INT,       &r, t, &v, NULL, pe); return r; }
long      shmem_long_fadd(long *t, long v, int pe)               { long      r; __shmem_amo(SHMEM_FADD, MPI_LONG,      &r, t, &v, NULL, pe); return r; }
long long shmem_longlong_fadd(long long *t, long long v, int pe) { long long r; __shmem_amo(SHMEM_FADD, MPI_LONG_LONG, &r, t, &v, NULL, pe); return r; }

/* 8.12: Atomic Memory fetch-and-operate Routines -- Fetch and Increment */
int       shmem_int_finc(int *t, int pe)             { int       v=1; int       r; __shmem_amo(SHMEM_FADD, MPI_INT,       &r, t, &v, NULL, pe); return r; }
long      shmem_long_finc(long *t, int pe)           { long      v=1; long      r; __shmem_amo(SHMEM_FADD, MPI_LONG,      &r, t, &v, NULL, pe); return r; }
long long shmem_longlong_finc(long long *t, int pe)  { long long v=1; long long r; __shmem_amo(SHMEM_FADD, MPI_LONG_LONG, &r, t, &v, NULL, pe); return r; }

/* 8.13: Atomic Memory Operation Routines -- Add */
void shmem_int_add(int *t, int v, int pe)                  { __shmem_amo(SHMEM_ADD, MPI_INT,       NULL, t, &v, NULL, pe); }
void shmem_long_add(long *t, long v, int pe)               { __shmem_amo(SHMEM_ADD, MPI_LONG,      NULL, t, &v, NULL, pe); }
void shmem_longlong_add(long long *t, long long v, int pe) { __shmem_amo(SHMEM_ADD, MPI_LONG_LONG, NULL, t, &v, NULL, pe); }

/* 8.13: Atomic Memory Operation Routines -- Increment */
void shmem_int_inc(int *t, int pe)            { int       v=1; __shmem_amo(SHMEM_ADD, MPI_INT,       NULL, t, &v, NULL, pe); }
void shmem_long_inc(long *t, int pe)          { long      v=1; __shmem_amo(SHMEM_ADD, MPI_LONG,      NULL, t, &v, NULL, pe); }
void shmem_longlong_inc(long long *t, int pe) { long long v=1; __shmem_amo(SHMEM_ADD, MPI_LONG_LONG, NULL, t, &v, NULL, pe); }

#if 0 
/* 8.14: Point-to-Point Synchronization Routines -- Wait */
void shmem_short_wait(short *var, short v);
void shmem_int_wait(int *var, int v);
void shmem_long_wait(long *var, long v);
void shmem_longlong_wait(long long *var, long long v);
void shmem_wait(long *ivar, long cmp_v);

/* 8.14: Point-to-Point Synchronization Routines -- Wait Until */
void shmem_short_wait_until(short *var, int c, short v);
void shmem_int_wait_until(int *var, int c, int v);
void shmem_long_wait_until(long *var, int c, long v);
void shmem_longlong_wait_until(long long *var, int c, long long v);
void shmem_wait_until(long *ivar, int cmp, long v);
#endif

/* TODO 
 * One might assume that the same subcomms are used more than once and thus caching these is prudent.
 */
static inline void __shmem_create_strided_comm(int pe_start, int log_pe_stride, int pe_size,       /* IN  */
                                               MPI_Comm * strided_comm, MPI_Group * strided_group) /* OUT */
{
    int * pe_list = malloc(pe_size*sizeof(int)); assert(pe_list);

    int pe_stride = 1<<log_pe_stride;
    for (int i=0; i<pe_size; i++)
        pe_list[i] = pe_start + i*pe_stride;

    MPI_Group_incl(SHMEM_GROUP_WORLD, pe_size, pe_list, strided_group);
    MPI_Comm_create_group(SHMEM_COMM_WORLD, *strided_group, 0 /* tag */, strided_comm); /* collective on group */

    free(pe_list);

    return;
}

static inline void __shmem_coll(enum shmem_coll_type_e coll, MPI_Datatype mpi_type, MPI_Op reduce_op,
                                void * target, const void * source, size_t len, 
                                int pe_root, int pe_start, int log_pe_stride, int pe_size)
{
    int collective_on_world = (pe_start==0 && log_pe_stride==0 && pe_size==shmem_mpi_size);

    MPI_Comm  strided_comm;
    MPI_Group strided_group;

    if (!collective_on_world)
        __shmem_create_strided_comm(pe_start, log_pe_stride, pe_size, &strided_comm, &strided_group);

    int count = 0;
    if ( likely(len<(size_t)INT32_MAX) ) {
        count = len;
    } else {
        /* TODO generate derived type ala BigMPI */
        __shmem_abort(coll);
    }

    switch (coll) {
        case SHMEM_BARRIER:
            MPI_Barrier( (collective_on_world==1) ? SHMEM_COMM_WORLD : strided_comm );
            break;
        case SHMEM_BROADCAST:
            {
                int bcast_root = pe_root;
                if (!collective_on_world) {
                    int world_ranks[1] = { pe_root };
                    int strided_ranks[1];
                    /* TODO I recall this function is expensive and further motivates caching of comm and such. */
                    MPI_Group_translate_ranks(SHMEM_GROUP_WORLD, 1, world_ranks, strided_group, strided_ranks);
                    bcast_root = strided_ranks[0];
                }
                if (pe_root==shmem_mpi_rank) {
                    int typesize;
                    MPI_Type_size(mpi_type, &typesize); /* could optimize away since only two cases possible */
                    memcpy(target, source, count*typesize);
                }
                MPI_Bcast(target, count, mpi_type, bcast_root, 
                          (collective_on_world==1) ? SHMEM_COMM_WORLD : strided_comm); 
            }
            break;
        case SHMEM_ALLGATHER:
            MPI_Allgather(source, count, mpi_type, target, count, mpi_type, 
                          (collective_on_world==1) ? SHMEM_COMM_WORLD : strided_comm);
            break;
        case SHMEM_ALLGATHERV:
            {
                int * rcounts = malloc(pe_size*sizeof(int)); assert(rcounts!=NULL);
                int * rdispls = malloc(pe_size*sizeof(int)); assert(rdispls!=NULL);
                MPI_Allgather(&count, 1, MPI_INT, rcounts, 1, MPI_INT, 
                              (collective_on_world==1) ? SHMEM_COMM_WORLD : strided_comm);
                rdispls[0] = 0;
                for (int i=1; i<pe_size; i++) {
                    rdispls[i] = rdispls[i-1] + rcounts[i-1];
                }
                MPI_Allgatherv(source, count, mpi_type, target, rcounts, rdispls, mpi_type, 
                               (collective_on_world==1) ? SHMEM_COMM_WORLD : strided_comm);
                free(rdispls);
                free(rcounts);
            }
            break;
#if 0
        case SHMEM_ALLREDUCE:
            MPI_Allreduce();
            break;
#endif
        default:
            __shmem_abort(coll);
            break;
    }

    if (!collective_on_world) {
        MPI_Group_free(&strided_group);
        MPI_Comm_free(&strided_comm);
    }
    return;
}

/* 8.15: Barrier Synchronization Routines */

void shmem_barrier(int PE_start, int logPE_stride, int PE_size, long *pSync)
{ 
    __shmem_coll(SHMEM_BARRIER, MPI_DATATYPE_NULL, MPI_OP_NULL, NULL, NULL, 0 /* count */, 0 /* root */,  PE_start, logPE_stride, PE_size); 
}

void shmem_barrier_all(void) 
{ 
    MPI_Barrier(SHMEM_COMM_WORLD); 
}

/* 8.18: Broadcast Routines */

void shmem_broadcast32(void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    __shmem_coll(SHMEM_BROADCAST, MPI_INT32_T, MPI_OP_NULL, target, source, nlong, PE_root, PE_start, logPE_stride, PE_size);
}
void shmem_broadcast64(void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    __shmem_coll(SHMEM_BROADCAST, MPI_INT64_T, MPI_OP_NULL, target, source, nlong, PE_root, PE_start, logPE_stride, PE_size);
}

/* 8.17: Collect Routines */

void shmem_collect32(void *target, const void *source, size_t nlong, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    __shmem_coll(SHMEM_ALLGATHERV, MPI_INT32_T, MPI_OP_NULL, target, source, nlong, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_collect64(void *target, const void *source, size_t nlong, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    __shmem_coll(SHMEM_ALLGATHERV, MPI_INT64_T, MPI_OP_NULL, target, source, nlong, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_fcollect32(void *target, const void *source, size_t nlong, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    __shmem_coll(SHMEM_ALLGATHER,  MPI_INT32_T, MPI_OP_NULL, target, source, nlong, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_fcollect64(void *target, const void *source, size_t nlong, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    __shmem_coll(SHMEM_ALLGATHER,  MPI_INT64_T, MPI_OP_NULL, target, source, nlong, 0 /* root */, PE_start, logPE_stride, PE_size);
}

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

#if 0

/* 8.19: Lock Routines */
void shmem_set_lock(long *lock);
void shmem_clear_lock(long *lock);
int  shmem_test_lock(long *lock);

#endif

/* A.1: Cache Management Routines (deprecated) */
void shmem_set_cache_inv(void) { return; }
void shmem_set_cache_line_inv(void *target) { return; }
void shmem_clear_cache_inv(void) { return; }
void shmem_clear_cache_line_inv(void *target) { return; }
void shmem_udcflush(void) { return; }
void shmem_udcflush_line(void *target) { return; }

/* Portals extensions */
double shmem_wtime(void) { return MPI_Wtime(); }
char* shmem_nodename(void)
{
    int namelen = 0;
    memset(shmem_procname, '\0', MPI_MAX_PROCESSOR_NAME);
    MPI_Get_processor_name( shmem_procname, &namelen );
    return shmem_procname;
}


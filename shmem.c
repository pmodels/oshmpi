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
#include "shmem-internals.h"
#include "shmem-wait.h"
#include "mcs-lock.h"

/* Mem-pool */
void bmem_free (void * ptr);
void * bmem_alloc (size_t size);
void * bmem_realloc (void * ptr, size_t size);
void * bmem_align (size_t alignment, size_t size);
/* Mem-pool */


void start_pes(int npes) 
{ 
    __shmem_initialize(); 
    atexit(__shmem_finalize);
    return;
}

/* 8.2: Query Routines */
int _num_pes(void) { return shmem_world_size; }
int shmem_n_pes(void) { return shmem_world_size; }
int _my_pe(void) { return shmem_world_rank; }
int shmem_my_pe(void) { return shmem_world_rank; }

/* 8.3: Accessibility Query Routines */
int shmem_pe_accessible(int pe) 
{ 
    return ( 0<=pe && pe<=shmem_world_size ); 
} 

int shmem_addr_accessible(void *addr, int pe) 
{ 
    if (0<=pe && pe<=shmem_world_size) {
        /* neither of these two variables is used here */
        enum shmem_window_id_e win_id;
        shmem_offset_t win_offset;
        /* __shmem_window_offset returns 0 on successful pointer lookup */
        return (0==__shmem_window_offset(addr, pe, &win_id, &win_offset));
    } else {
        return 0;
    }
}

/* 8.4: Symmetric Heap Routines */

void *shmemalign(size_t alignment, size_t size)
{
#if SHEAP_HACK > 1	
    size_t align_bump = (size%alignment ? 1 : 0);
    size_t align_size = (size/alignment + align_bump) * alignment;

#if SHMEM_DEBUG > 1
    printf("[%d] size       = %zu alignment  = %zu \n", shmem_world_rank, size, alignment );
    printf("[%d] align_size = %zu align_bump = %zu \n", shmem_world_rank, align_size, align_bump );
    printf("[%d] shmem_sheap_current_ptr = %p  \n", shmem_world_rank, shmem_sheap_current_ptr );
    fflush(stdout);
#endif

    /* this is the hack-tastic version so no check for overrun */
    void * address = shmem_sheap_current_ptr;
    shmem_sheap_current_ptr += align_size;

#if SHMEM_DEBUG > 1
    printf("[%d] shmemalign/shmalloc is going to return address = %p  \n", shmem_world_rank, address );
    fflush(stdout);
#endif

    return address;
#else
    return bmem_align (alignment, size);
#endif
}

void *shmalloc(size_t size)
{
    /* use page size for debugging purposes */
#if SHEAP_HACK > 1
    if (shmem_world_rank == 0) { printf("Using hack-tastic version of sheap\n"); }	
    const int default_alignment = 4096;
    return shmemalign(default_alignment, size);
#else    
    return bmem_alloc (size);
#endif
}

void *shrealloc(void *ptr, size_t size)
{
#if SHEAP_HACK > 1	
    __shmem_abort(size, "shrealloc is not implemented in the hack-tastic version of sheap");
    return NULL;
#else    
   return bmem_realloc (ptr, size);
#endif
}

void shfree(void *ptr)
{
    	
#if SHEAP_HACK > 1	
    __shmem_warn("shfree is a no-op in the hack-tastic version of sheap");
    return;
#else    
    return bmem_free (ptr);
#endif
}

void shmem_quiet(void)
{
    /* The Portals4 interpretation of quiet is 
     * "remote completion of all pending events",
     * which I take to mean remote completion of RMA. */
    __shmem_remote_sync();
    __shmem_local_sync();
}

void shmem_fence(void)
{
    /* Doing fence as quiet is scalable; the per-rank method is not. 
     *  - Keith Underwood on OpenSHMEM list */
    __shmem_remote_sync();
    __shmem_local_sync();
}

/* 8.5: Remote Pointer Operations */
void *shmem_ptr(void *target, int pe)
{ 
#ifdef USE_SMP_OPTIMIZATIONS
    if (shmem_world_is_smp) {
        /* TODO shared memory window optimization */
        __shmem_abort(pe, "intranode shared memory pointer access not implemented");
        return NULL; 
    } else 
#endif
    {
        return (pe==shmem_world_rank ? target : NULL);
    }
}

/* 8.6: Elemental Put Routines */
void shmem_float_p(float *addr, float v, int pe)
{
    __shmem_put(MPI_FLOAT, addr, &v, 1, pe);
}
void shmem_double_p(double *addr, double v, int pe)
{
    __shmem_put(MPI_DOUBLE, addr, &v, 1, pe);
}
void shmem_longdouble_p(long double *addr, long double v, int pe)
{
    __shmem_put(MPI_LONG_DOUBLE, addr, &v, 1, pe);
}
void shmem_char_p(char *addr, char v, int pe)
{
    __shmem_put(MPI_CHAR, addr, &v, 1, pe);
}
void shmem_short_p(short *addr, short v, int pe)
{
    __shmem_put(MPI_SHORT, addr, &v, 1, pe);
}
void shmem_int_p(int *addr, int v, int pe)
{
    __shmem_put(MPI_INT, addr, &v, 1, pe);
}
void shmem_long_p(long *addr, long v, int pe)
{
    __shmem_put(MPI_LONG, addr, &v, 1, pe);
}
void shmem_longlong_p(long long *addr, long long v, int pe)
{
    __shmem_put(MPI_LONG_LONG, addr, &v, 1, pe);
}

/* 8.7: Block Data Put Routines */
void shmem_float_put(float *target, const float *source, size_t len, int pe)
{
    __shmem_put(MPI_FLOAT, target, source, len, pe);
}
void shmem_double_put(double *target, const double *source, size_t len, int pe)
{
    __shmem_put(MPI_DOUBLE, target, source, len, pe);
}
void shmem_longdouble_put(long double *target, const long double *source, size_t len, int pe)
{
    __shmem_put(MPI_LONG_DOUBLE, target, source, len, pe);
}
void shmem_char_put(char *target, const char *source, size_t len, int pe)
{
    __shmem_put(MPI_CHAR, target, source, len, pe);
}
void shmem_short_put(short *target, const short *source, size_t len, int pe)
{
    __shmem_put(MPI_SHORT, target, source, len, pe);
}
void shmem_int_put(int *target, const int *source, size_t len, int pe)
{
    __shmem_put(MPI_INT, target, source, len, pe);
}
void shmem_long_put(long *target, const long *source, size_t len, int pe)
{
    __shmem_put(MPI_LONG, target, source, len, pe);
}
void shmem_longlong_put(long long *target, const long long *source, size_t len, int pe)
{
    __shmem_put(MPI_LONG_LONG, target, source, len, pe);
}
void shmem_putmem(void *target, const void *source, size_t len, int pe)
{
    __shmem_put(MPI_BYTE, target, source, len, pe);
}
void shmem_put32(void *target, const void *source, size_t len, int pe)
{
    __shmem_put(MPI_INT32_T, target, source, len, pe);
}
void shmem_put64(void *target, const void *source, size_t len, int pe)
{
    __shmem_put(MPI_DOUBLE, target, source, len, pe);
}
void shmem_put128(void *target, const void *source, size_t len, int pe)
{
    __shmem_put(MPI_C_DOUBLE_COMPLEX, target, source, len, pe);
}
void shmem_complexf_put(float complex * target, const float complex * source, size_t len, int pe)
{
    __shmem_put(MPI_COMPLEX, target, source, len, pe);
}
void shmem_complexd_put(double complex * target, const double complex * source, size_t len, int pe)
{
    __shmem_put(MPI_DOUBLE_COMPLEX, target, source, len, pe);
}

/* 8.9: Elemental Data Get Routines */
float shmem_float_g(float *addr, int pe)
{
    float v;
    __shmem_get(MPI_FLOAT, &v, addr, 1, pe);
    return v;
}
double shmem_double_g(double *addr, int pe)
{
    double v;
    __shmem_get(MPI_DOUBLE, &v, addr, 1, pe);
    return v;
}
long double shmem_longdouble_g(long double *addr, int pe)
{
    long double v;
    __shmem_get(MPI_LONG_DOUBLE, &v, addr, 1, pe);
    return v;
}
char shmem_char_g(char *addr, int pe)
{
    char v;
    __shmem_get(MPI_CHAR, &v, addr, 1, pe);
    return v;
}
short shmem_short_g(short *addr, int pe)
{
    short v;
    __shmem_get(MPI_SHORT, &v, addr, 1, pe);
    return v;
}
int shmem_int_g(int *addr, int pe)
{
    int v;
    __shmem_get(MPI_INT, &v, addr, 1, pe);
    return v;
}
long shmem_long_g(long *addr, int pe)
{
    long v;
    __shmem_get(MPI_LONG, &v, addr, 1, pe);
    return v;
}
long long shmem_longlong_g(long long *addr, int pe)
{
    long long v;
    __shmem_get(MPI_LONG_LONG, &v, addr, 1, pe);
    return v;
}

/* 8.10 Block Data Get Routines */
void shmem_float_get(float *target, const float *source, size_t len, int pe)
{
    __shmem_get(MPI_FLOAT, target, source, len, pe);
}
void shmem_double_get(double *target, const double *source, size_t len, int pe)
{
    __shmem_get(MPI_DOUBLE, target, source, len, pe);
}
void shmem_longdouble_get(long double *target, const long double *source, size_t len, int pe)
{
    __shmem_get(MPI_LONG_DOUBLE, target, source, len, pe);
}
void shmem_char_get(char *target, const char *source, size_t len, int pe)
{
    __shmem_get(MPI_CHAR, target, source, len, pe);
}
void shmem_short_get(short *target, const short *source, size_t len, int pe)
{
    __shmem_get(MPI_SHORT, target, source, len, pe);
}
void shmem_int_get(int *target, const int *source, size_t len, int pe)
{
    __shmem_get(MPI_INT, target, source, len, pe);
}
void shmem_long_get(long *target, const long *source, size_t len, int pe)
{
    __shmem_get(MPI_LONG, target, source, len, pe);
}
void shmem_longlong_get(long long *target, const long long *source, size_t len, int pe)
{
    __shmem_get(MPI_LONG_LONG, target, source, len, pe);
}
void shmem_getmem(void *target, const void *source, size_t len, int pe)
{
    __shmem_get(MPI_BYTE, target, source, len, pe);
}
void shmem_get32(void *target, const void *source, size_t len, int pe)
{
    __shmem_get(MPI_INT32_T, target, source, len, pe);
}
void shmem_get64(void *target, const void *source, size_t len, int pe)
{
    __shmem_get(MPI_DOUBLE, target, source, len, pe);
}
void shmem_get128(void *target, const void *source, size_t len, int pe)
{
    __shmem_get(MPI_C_DOUBLE_COMPLEX, target, source, len, pe);
}
void shmem_complexf_get(float complex * target, const float complex * source, size_t len, int pe)
{
    __shmem_get(MPI_COMPLEX, target, source, len, pe);
}
void shmem_complexd_get(double complex * target, const double complex * source, size_t len, int pe)
{
    __shmem_get(MPI_DOUBLE_COMPLEX, target, source, len, pe);
}

/* 8.8: Strided Put Routines */
void shmem_float_iput(float *target, const float *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_put_strided(MPI_FLOAT, target, source, tst, sst, len, pe);
}
void shmem_double_iput(double *target, const double *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_put_strided(MPI_DOUBLE, target, source, tst, sst, len, pe);
}
void shmem_longdouble_iput(long double *target, const long double *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_put_strided(MPI_LONG_DOUBLE, target, source, tst, sst, len, pe);
}
void shmem_short_iput(short *target, const short *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_put_strided(MPI_SHORT, target, source, tst, sst, len, pe);
}
void shmem_int_iput(int *target, const int *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_put_strided(MPI_INT, target, source, tst, sst, len, pe);
}
void shmem_long_iput(long *target, const long *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_put_strided(MPI_LONG, target, source, tst, sst, len, pe);
}
void shmem_longlong_iput(long long *target, const long long *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_put_strided(MPI_LONG_LONG, target, source, tst, sst, len, pe);
}
void shmem_iput32(void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_put_strided(MPI_INT32_T, target, source, tst, sst, len, pe);
}
void shmem_iput64(void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_put_strided(MPI_DOUBLE, target, source, tst, sst, len, pe);
}
void shmem_iput128(void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_put_strided(MPI_C_DOUBLE_COMPLEX, target, source, tst, sst, len, pe);
}

/* 8.11: Strided Get Routines */
void shmem_float_iget(float *target, const float *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_get_strided(MPI_FLOAT, target, source, tst, sst, len, pe);
}
void shmem_double_iget(double *target, const double *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_get_strided(MPI_DOUBLE, target, source, tst, sst, len, pe);
}
void shmem_longdouble_iget(long double *target, const long double *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_get_strided(MPI_LONG_DOUBLE, target, source, tst, sst, len, pe);
}
void shmem_short_iget(short *target, const short *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_get_strided(MPI_SHORT, target, source, tst, sst, len, pe);
}
void shmem_int_iget(int *target, const int *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_get_strided(MPI_INT, target, source, tst, sst, len, pe);
}
void shmem_long_iget(long *target, const long *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_get_strided(MPI_LONG, target, source, tst, sst, len, pe);
}
void shmem_longlong_iget(long long *target, const long long *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_get_strided(MPI_LONG_LONG, target, source, tst, sst, len, pe);
}
void shmem_iget32(void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_get_strided(MPI_INT32_T, target, source, tst, sst, len, pe);
}
void shmem_iget64(void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_get_strided(MPI_DOUBLE, target, source, tst, sst, len, pe);
}
void shmem_iget128(void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe)
{
    __shmem_get_strided(MPI_C_DOUBLE_COMPLEX, target, source, tst, sst, len, pe);
}

/* Naming conventions for shorthand:
 * r = return v
 * c = comparand
 * v = v (input)
 * t = target 
 */

/* 8.12: Atomic Memory fetch-and-operate Routines -- Swap */
float shmem_float_swap(float *t, float v, int pe)            
{ 
    float r; 
    __shmem_swap(MPI_FLOAT, &r, t, &v, pe) ; 
    return r; 
}
double shmem_double_swap(double *t, double v, int pe)         
{ 
    double r; 
    __shmem_swap(MPI_DOUBLE, &r, t, &v, pe) ; 
    return r; 
}
int shmem_int_swap(int *t, int v, int pe)                  
{ 
    int r; 
    __shmem_swap(MPI_INT, &r, t, &v, pe) ; 
    return r; 
}
long shmem_long_swap(long *t, long v, int pe)               
{ 
    long r; 
    __shmem_swap(MPI_LONG, &r, t, &v, pe) ; 
    return r; 
}
long long shmem_longlong_swap(long long *t, long long v, int pe) 
{ 
    long long r; 
    __shmem_swap(MPI_LONG_LONG, &r, t, &v, pe) ; 
    return r; 
}
long shmem_swap(long *t, long v, int pe)                    
{ 
    long r; 
    __shmem_swap(MPI_LONG, &r, t, &v, pe) ; 
    return r; 
}

/* 8.12: Atomic Memory fetch-and-operate Routines -- Cswap */
int shmem_int_cswap(int *t, int c, int v, int pe)                         
{ 
    int r; 
    __shmem_cswap(MPI_INT, &r, t, &v, &c, pe) ; 
    return r; 
}
long shmem_long_cswap(long *t, long c, long v, int pe)                     
{ 
    long r; 
    __shmem_cswap(MPI_LONG, &r, t, &v, &c, pe) ; 
    return r; 
}
long long shmem_longlong_cswap(long long * t, long long c, long long v, int pe) 
{ 
    long long r; 
    __shmem_cswap(MPI_LONG_LONG, &r, t, &v, &c, pe) ; 
    return r; 
}

/* 8.12: Atomic Memory fetch-and-operate Routines -- Fetch and Add */
int shmem_int_fadd(int *t, int v, int pe)                  
{ 
    int r; 
    __shmem_fadd(MPI_INT, &r, t, &v, pe); 
    return r; 
}
long shmem_long_fadd(long *t, long v, int pe)               
{ 
    long r; 
    __shmem_fadd(MPI_LONG, &r, t, &v, pe); 
    return r; 
}
long long shmem_longlong_fadd(long long *t, long long v, int pe) 
{ 
    long long r; 
    __shmem_fadd(MPI_LONG_LONG, &r, t, &v, pe); 
    return r; 
}

/* 8.12: Atomic Memory fetch-and-operate Routines -- Fetch and Increment */
int shmem_int_finc(int *t, int pe)             
{ 
    int r, v=1; 
    __shmem_fadd(MPI_INT, &r, t, &v, pe); 
    return r; 
}
long shmem_long_finc(long *t, int pe)           
{ 
    long r, v=1; 
    __shmem_fadd(MPI_LONG, &r, t, &v, pe); 
    return r; 
}
long long shmem_longlong_finc(long long *t, int pe)  
{ 
    long long r, v=1; 
    __shmem_fadd(MPI_LONG_LONG, &r, t, &v, pe); 
    return r; 
}

/* 8.13: Atomic Memory Operation Routines -- Add */
void shmem_int_add(int *t, int v, int pe)                  
{ 
    __shmem_add(MPI_INT, t, &v, pe); 
}
void shmem_long_add(long *t, long v, int pe)               
{ 
    __shmem_add(MPI_LONG, t, &v, pe); 
}
void shmem_longlong_add(long long *t, long long v, int pe) 
{ 
    __shmem_add(MPI_LONG_LONG, t, &v, pe); 
}

/* 8.13: Atomic Memory Operation Routines -- Increment */
void shmem_int_inc(int *t, int pe)            
{ 
    int v=1;
    __shmem_add(MPI_INT, t, &v, pe); 
}
void shmem_long_inc(long *t, int pe)          
{ 
    long v=1;
    __shmem_add(MPI_LONG, t, &v, pe); 
}
void shmem_longlong_inc(long long *t, int pe) 
{ 
    long long v=1;
    __shmem_add(MPI_LONG_LONG, t, &v, pe); 
}

/* 8.14: Point-to-Point Synchronization Routines -- Wait */
void shmem_short_wait(short *var, short v) { SHMEM_WAIT(var,v); }
void shmem_int_wait(int *var, int v) { SHMEM_WAIT(var,v); }
void shmem_long_wait(long *var, long v) { SHMEM_WAIT(var,v); }
void shmem_longlong_wait(long long *var, long long v) { SHMEM_WAIT(var,v); }
void shmem_wait(long *var, long v) { SHMEM_WAIT(var,v); }

/* 8.14: Point-to-Point Synchronization Routines -- Wait Until */
void shmem_short_wait_until(short *var, int c, short v) { SHMEM_WAIT_UNTIL(var, c, v); }
void shmem_int_wait_until(int *var, int c, int v) { SHMEM_WAIT_UNTIL(var, c, v); }
void shmem_long_wait_until(long *var, int c, long v) { SHMEM_WAIT_UNTIL(var, c, v); }
void shmem_longlong_wait_until(long long *var, int c, long long v) { SHMEM_WAIT_UNTIL(var, c, v); }
void shmem_wait_until(long *var, int c, long v) { SHMEM_WAIT_UNTIL(var, c, v); }

/* 8.15: Barrier Synchronization Routines */

void shmem_barrier(int PE_start, int logPE_stride, int PE_size, long *pSync)
{ 
    __shmem_remote_sync();
    __shmem_local_sync();
    __shmem_set_psync(_SHMEM_BARRIER_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_BARRIER, MPI_DATATYPE_NULL, MPI_OP_NULL, NULL, NULL, 0 /* count */, 0 /* root */,  PE_start, logPE_stride, PE_size); 
}

void shmem_barrier_all(void) 
{ 
    __shmem_remote_sync();
    __shmem_local_sync();
    MPI_Barrier(SHMEM_COMM_WORLD); 
    //__shmem_coll(SHMEM_BARRIER, MPI_DATATYPE_NULL, MPI_OP_NULL, NULL, NULL, 0 /* count */, 0 /* root */, 0, 0, shmem_world_size ); 
}

/* 8.18: Broadcast Routines */

void shmem_broadcast32(void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    __shmem_set_psync(_SHMEM_BCAST_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_BROADCAST, MPI_INT32_T, MPI_OP_NULL, target, source, nlong, PE_root, PE_start, logPE_stride, PE_size);
}
void shmem_broadcast64(void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    __shmem_set_psync(_SHMEM_BCAST_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_BROADCAST, MPI_INT64_T, MPI_OP_NULL, target, source, nlong, PE_root, PE_start, logPE_stride, PE_size);
}

/* 8.17: Collect Routines */

void shmem_collect32(void *target, const void *source, size_t nlong, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    __shmem_set_psync(_SHMEM_COLLECT_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLGATHERV, MPI_INT32_T, MPI_OP_NULL, target, source, nlong, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_collect64(void *target, const void *source, size_t nlong, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    __shmem_set_psync(_SHMEM_COLLECT_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLGATHERV, MPI_INT64_T, MPI_OP_NULL, target, source, nlong, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_fcollect32(void *target, const void *source, size_t nlong, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    __shmem_set_psync(_SHMEM_COLLECT_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLGATHER,  MPI_INT32_T, MPI_OP_NULL, target, source, nlong, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_fcollect64(void *target, const void *source, size_t nlong, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
    __shmem_set_psync(_SHMEM_COLLECT_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLGATHER,  MPI_INT64_T, MPI_OP_NULL, target, source, nlong, 0 /* root */, PE_start, logPE_stride, PE_size);
}

/* 8.16: Reduction Routines */

void shmem_short_and_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_SHORT, MPI_LAND, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_int_and_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_INT, MPI_LAND, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_long_and_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG, MPI_LAND, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longlong_and_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_LONG, MPI_LAND, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}

void shmem_short_or_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_SHORT, MPI_BOR, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_int_or_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_INT, MPI_BOR, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_long_or_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG, MPI_BOR, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longlong_or_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_LONG, MPI_BOR, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}

void shmem_short_xor_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_SHORT, MPI_BXOR, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_int_xor_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_INT, MPI_BXOR, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_long_xor_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG, MPI_BXOR, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longlong_xor_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_LONG, MPI_BXOR, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}

void shmem_float_min_to_all(float *target, float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_FLOAT, MPI_MIN, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_double_min_to_all(double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_DOUBLE, MPI_MIN, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longdouble_min_to_all(long double *target, long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_DOUBLE, MPI_MIN, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_short_min_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_SHORT, MPI_MIN, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_int_min_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_INT, MPI_MIN, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_long_min_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG, MPI_MIN, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longlong_min_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_LONG, MPI_MIN, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}

void shmem_float_max_to_all(float *target, float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_FLOAT, MPI_MAX, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_double_max_to_all(double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_DOUBLE, MPI_MAX, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longdouble_max_to_all(long double *target, long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_DOUBLE, MPI_MAX, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_short_max_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_SHORT, MPI_MAX, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_int_max_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_INT, MPI_MAX, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_long_max_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG, MPI_MAX, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longlong_max_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_LONG, MPI_MAX, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}

void shmem_float_sum_to_all(float *target, float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_FLOAT, MPI_SUM, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_double_sum_to_all(double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_DOUBLE, MPI_SUM, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longdouble_sum_to_all(long double *target, long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_DOUBLE, MPI_SUM, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_short_sum_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_SHORT, MPI_SUM, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_int_sum_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_INT, MPI_SUM, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_long_sum_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG, MPI_SUM, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longlong_sum_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_LONG, MPI_SUM, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}

void shmem_float_prod_to_all(float *target, float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_FLOAT, MPI_PROD, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_double_prod_to_all(double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_DOUBLE, MPI_PROD, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longdouble_prod_to_all(long double *target, long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_DOUBLE, MPI_PROD, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_short_prod_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_SHORT, MPI_PROD, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_int_prod_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_INT, MPI_PROD, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_long_prod_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG, MPI_PROD, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longlong_prod_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
    __shmem_set_psync(_SHMEM_REDUCE_SYNC_SIZE, _SHMEM_SYNC_VALUE, pSync);
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_LONG, MPI_PROD, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}

/* 8.19: Lock Routines */
void shmem_set_lock(long *lock)
{
	acquire_mcslock(lock); 
	return;
}

void shmem_clear_lock(long *lock)
{
	release_mcslock(lock); 
	return;
}

int  shmem_test_lock(long *lock)
{
	int success;
	
	test_mcslock(lock, &success);

	return success;
}

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
    /* In general, nodename != procname, of course, but there are 
     * many implementations where this will be true because the 
     * procname is just the IP address. */
    int namelen = 0;
    memset(shmem_procname, '\0', MPI_MAX_PROCESSOR_NAME);
    MPI_Get_processor_name( shmem_procname, &namelen );
    return shmem_procname;
}


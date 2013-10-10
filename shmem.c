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

/* configuration settings */
/* This is the only support mode right now. */
#define USE_ORDERED_RMA
/* This should always be set unless your MPI sucks. */
#define USE_ALLREDUCE
/* Not implemented yet. */
//#define USE_SMP_OPTIMIZATIONS

#if ( defined(__GNUC__) && (__GNUC__ >= 3) ) || defined(__IBMC__) || defined(__INTEL_COMPILER) || defined(__clang__)
#  define unlikely(x_) __builtin_expect(!!(x_),0)
#  define likely(x_)   __builtin_expect(!!(x_),1)
#else
#  define unlikely(x_) (x_)
#  define likely(x_)   (x_)
#endif

/* this code deals with SHMEM communication out of symmetric but non-heap data */
#if defined(__APPLE__)
    /* https://developer.apple.com/library/mac//documentation/Darwin/Reference/ManPages/10.7/man3/end.3.html */
#include <mach-o/getsect.h>
    unsigned long get_end();
    unsigned long get_etext();
    unsigned long get_edata();
#elif defined(_AIX)
#warning AIX is completely untested.
    /* http://pic.dhe.ibm.com/infocenter/aix/v6r1/topic/com.ibm.aix.basetechref/doc/basetrf1/_end.htm */
    extern _end;
    extern _etext;
    extern _edata;
    unsigned long get_end()   { return &_end;   }
    unsigned long get_etext() { return &_etext; }
    unsigned long get_edata() { return &_edata; }
#elif defined(__linux__)
    /* http://man7.org/linux/man-pages/man3/end.3.html */
    extern char data_start;
    extern char etext;
    extern char edata;
    extern char end;
    unsigned long get_sdata() { return (unsigned long)&data_start;   }
    unsigned long get_end()   { return (unsigned long)&end;   }
    unsigned long get_etext() { return (unsigned long)&etext; }
    unsigned long get_edata() { return (unsigned long)&edata; }
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
      defined(__bsdi__) || defined(__DragonFly__)  // Known BSD variants
#  error BSD is not supported yet.
#elif defined(__bgq__)
#  error Blue Gene/Q is not supported yet.
#else
#  error Unknown and unsupported operating system.
#endif

/*****************************************************************/
/* TODO convert all the global status into a struct ala ARMCI-MPI */
/* requires TLS if MPI is thread-based */
static MPI_Comm  SHMEM_COMM_WORLD;
static MPI_Group SHMEM_GROUP_WORLD; /* used for creating logpe comms */

static int       shmem_is_initialized = 0;
static int       shmem_is_finalized   = 0;
static int       shmem_world_size, shmem_world_rank;
static char      shmem_procname[MPI_MAX_PROCESSOR_NAME];

#ifdef USE_SMP_OPTIMIZATIONS
static MPI_Comm  SHMEM_COMM_NODE;
static MPI_Group SHMEM_GROUP_NODE; /* may not be needed as global */
static int       shmem_world_is_smp   = 0;
static int       shmem_node_size, shmem_node_rank;
static int *     shmem_smp_rank_list;
#endif

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
static void *  shmem_sheap_current_ptr;
/*****************************************************************/

enum shmem_window_id_e { SHMEM_SHEAP_WINDOW = 0, SHMEM_ETEXT_WINDOW = 1, SHMEM_INVALID_WINDOW = -1 };
enum shmem_rma_type_e  { SHMEM_PUT = 0, SHMEM_GET = 1, SHMEM_IPUT = 2, SHMEM_IGET = 4};
enum shmem_amo_type_e  { SHMEM_SWAP = 0, SHMEM_CSWAP = 1, SHMEM_ADD = 2, SHMEM_FADD = 4};
enum shmem_coll_type_e { SHMEM_BARRIER = 0, SHMEM_BROADCAST = 1, SHMEM_ALLREDUCE = 2, SHMEM_ALLGATHER = 4, SHMEM_ALLGATHERV = 8};

static void __shmem_warn(char * message)
{
#if SHMEM_DEBUG > 0
    printf("[%d] %s \n", shmem_world_rank, message);
    fflush(stdout);
#endif
    return;
}

static void __shmem_abort(int code, char * message)
{
    printf("[%d] %s \n", shmem_world_rank, message);
    fflush(stdout);
    MPI_Abort(SHMEM_COMM_WORLD, code);
    return;
}

/* 8.1: Initialization Routines */
static int __shmem_address_is_symmetric(size_t my_sheap_base_ptr)
{
    /* I am not sure if there is a better way to operate on addresses... */
    /* cannot fuse allreduces because max{base,-base} trick does not work for unsigned */

    int is_symmetric = 0;
    size_t minbase = 0;
    size_t maxbase = 0;

    /* The latter might be faster on machines with bad collective implementations. 
     * On Blue Gene, Allreduce is definitely the way to go. 
     */
#ifdef USE_ALLREDUCE
    MPI_Allreduce( &my_sheap_base_ptr, &minbase, 1, sizeof(size_t)==4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG, MPI_MIN, SHMEM_COMM_WORLD );
    MPI_Allreduce( &my_sheap_base_ptr, &maxbase, 1, sizeof(size_t)==4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG, MPI_MAX, SHMEM_COMM_WORLD );
    is_symmetric = ((minbase==my_sheap_base_ptr && my_sheap_base_ptr==maxbase) ? 1 : 0);
#else
    MPI_Reduce( &my_sheap_base_ptr, &minbase, 1, sizeof(size_t)==4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG, MPI_MIN, 0, SHMEM_COMM_WORLD );
    MPI_Reduce( &my_sheap_base_ptr, &maxbase, 1, sizeof(size_t)==4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG, MPI_MAX, 0, SHMEM_COMM_WORLD );
    if (shmem_world_rank==0)
        is_symmetric = ((minbase==my_sheap_base_ptr && my_sheap_base_ptr==maxbase) ? 1 : 0);
    MPI_Bcast( &is_symmetric, 1, MPI_INT, 0, SHMEM_COMM_WORLD );
#endif
    return is_symmetric;
}

static void __shmem_initialize(void)
{
    int flag, provided;
    MPI_Initialized(&flag);
    if (!flag) 
        MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &provided);

    if (!shmem_is_initialized) {

        MPI_Comm_dup(MPI_COMM_WORLD, &SHMEM_COMM_WORLD);
        MPI_Comm_size(SHMEM_COMM_WORLD, &shmem_world_size);
        MPI_Comm_rank(SHMEM_COMM_WORLD, &shmem_world_rank);
        MPI_Comm_group(SHMEM_COMM_WORLD, &SHMEM_GROUP_WORLD);

#ifdef USE_SMP_OPTIMIZATIONS
        {
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0 /* key */, MPI_INFO_NULL, &SHMEM_COMM_NODE);
            MPI_Comm_size(SHMEM_COMM_NODE, &shmem_node_size);
            MPI_Comm_rank(SHMEM_COMM_NODE, &shmem_node_rank);
            MPI_Comm_group(SHMEM_COMM_NODE, &SHMEM_GROUP_NODE);

            int result;
            MPI_Comm_compare(SHMEM_COMM_WORLD, SHMEM_COMM_NODE, &result);
            shmem_world_is_smp = (result==MPI_IDENT || result==MPI_CONGRUENT) ? 1 : 0;

            shmem_smp_rank_list  = (int*) malloc( shmem_node_size*sizeof(int) );
            int * temp_rank_list = (int*) malloc( shmem_node_size*sizeof(int) );
            for (int i=0; i<shmem_node_size; i++) {
                temp_rank_list[i] = i;
            }
            /* translate ranks in the node group to world ranks */
            MPI_Group_translate_ranks(SHMEM_GROUP_NODE,  shmem_node_size, temp_rank_list, 
                                      SHMEM_GROUP_WORLD, shmem_smp_rank_list);
            free(temp_rank_list);
        }
#endif

        {
            char * c = getenv("SHMEM_SYMMETRIC_HEAP_SIZE");
            shmem_sheap_size = ( (c) ? atoi(c) : 128*1024*1024 );
        }

#ifdef USE_ORDERED_RMA
        MPI_Info info = MPI_INFO_NULL;
#else
#warning NOT READY
        /* TODO 
         * Set info keys to disable unnecessary accumulate support. */
        MPI_Info info = MPI_INFO_NULL; 
#endif

        void * my_sheap_base_ptr = NULL;

#ifdef USE_SMP_OPTIMIZATIONS
        if (shmem_world_is_smp) {
            MPI_Win_allocate_shared((MPI_Aint)shmem_sheap_size, 1 /* disp_unit */, info, SHMEM_COMM_WORLD, &my_sheap_base_ptr, &shmem_sheap_win);
        } else 
#endif
        {
            MPI_Win_allocate((MPI_Aint)shmem_sheap_size, 1 /* disp_unit */, info, SHMEM_COMM_WORLD, &my_sheap_base_ptr, &shmem_sheap_win);
        }
        MPI_Win_lock_all(0, shmem_sheap_win);

        /* TODO
         * Even if world is not an SMP, we can still leverage shared-memory within the node.
         * However, this may lead to undefined behavior when overlapping windows are accessed
         * by load-store and by RMA at the same time.
         */

        shmem_sheap_is_symmetric = __shmem_address_is_symmetric((size_t)my_sheap_base_ptr);
#if SHMEM_DEBUG > 0
        if (shmem_world_rank==0) {
            printf("[*] sheap %s symmetric \n", shmem_sheap_is_symmetric==1 ? "is" : "is not");
            fflush(stdout);
        }
#endif

        if (!shmem_sheap_is_symmetric) {
            /* non-symmetric heap requires O(nproc) metadata */
            shmem_sheap_base_ptrs = malloc(shmem_world_size*sizeof(void*)); assert(shmem_sheap_base_ptrs!=NULL);
            MPI_Allgather(&my_sheap_base_ptr, sizeof(void*), MPI_BYTE, shmem_sheap_base_ptrs, sizeof(void*), MPI_BYTE, SHMEM_COMM_WORLD);
        }
        shmem_sheap_mybase_ptr = my_sheap_base_ptr; /* use as shortcut for local lookup when non-symmetric */

        /* this is the hack-tastic sheap initialization */
        shmem_sheap_current_ptr = shmem_sheap_mybase_ptr;
#if SHMEM_DEBUG > 1
        printf("[%d] shmem_sheap_current_ptr  = %p  \n", shmem_world_rank, shmem_sheap_current_ptr );
        fflush(stdout);
#endif

        /* FIXME eliminate platform-specific stuff here i.e. find a way to move to top */
#ifdef __APPLE__
	void *        my_etext_base_ptr = (void*) get_etext();
#else
	void *        my_etext_base_ptr = (void *)get_sdata();
#endif
        unsigned long long_etext_size   = get_end() - (unsigned long)my_etext_base_ptr;
        assert(long_etext_size<(unsigned long)INT32_MAX); 
        shmem_etext_size = (int)long_etext_size;
#if SHMEM_DEBUG > 5
        printf("[%d] get_etext()       = %p \n", shmem_world_rank, (void*)get_etext() );
        printf("[%d] get_edata()       = %p \n", shmem_world_rank, (void*)get_edata() );
        printf("[%d] get_end()         = %p \n", shmem_world_rank, (void*)get_end()   );
        //printf("[%d] long_etext_size   = %lu \n", shmem_world_rank, long_etext_size );
        printf("[%d] shmem_etext_size  = %d  \n", shmem_world_rank, shmem_etext_size );
        //printf("[%d] my_etext_base_ptr = %p  \n", shmem_world_rank, my_etext_base_ptr );
        fflush(stdout);
#endif

        MPI_Win_create(my_etext_base_ptr, shmem_etext_size, 1 /* disp_unit */, MPI_INFO_NULL, SHMEM_COMM_WORLD, &shmem_etext_win);
        MPI_Win_lock_all(0, shmem_etext_win);

        shmem_etext_is_symmetric = __shmem_address_is_symmetric((size_t)my_etext_base_ptr);
#if SHMEM_DEBUG > 0
        if (shmem_world_rank==0) {
            printf("[*] etext %s symmetric \n", shmem_etext_is_symmetric==1 ? "is" : "is not");
            fflush(stdout);
        }
#endif

        if (!shmem_etext_is_symmetric) {
            /* non-symmetric heap requires O(nproc) metadata */
            shmem_etext_base_ptrs = malloc(shmem_world_size*sizeof(void*)); assert(shmem_etext_base_ptrs!=NULL);
            MPI_Allgather(&my_etext_base_ptr, sizeof(void*), MPI_BYTE, shmem_etext_base_ptrs, sizeof(void*), MPI_BYTE, SHMEM_COMM_WORLD);
        } else {
            shmem_etext_mybase_ptr = my_etext_base_ptr;
        }

        { /* It is hard if not impossible to implement SHMEM without the UNIFIED model. */
            int   sheap_flag = 0;
            int * sheap_model = NULL;
            MPI_Win_get_attr(shmem_sheap_win, MPI_WIN_MODEL, &sheap_model, &sheap_flag);
            int   etext_flag = 0;
            int * etext_model = NULL;
            MPI_Win_get_attr(shmem_etext_win, MPI_WIN_MODEL, &etext_model, &etext_flag);
            /*
	    if (*sheap_model != MPI_WIN_UNIFIED || *etext_model != MPI_WIN_UNIFIED) {
                __shmem_abort(1, "You cannot use this implementation of SHMEM without the UNIFIED model.\n");
            }
	    */
        }

        shmem_is_initialized = 1;
    }
    return;
}

static void __shmem_finalize(void)
{
    int flag;
    MPI_Finalized(&flag);

    if (!flag) {
        if (shmem_is_initialized && !shmem_is_finalized) {
            if (!shmem_sheap_is_symmetric) 
                free(shmem_sheap_base_ptrs);
            if (!shmem_etext_is_symmetric) 
                free(shmem_etext_base_ptrs);

            MPI_Win_unlock_all(shmem_etext_win);
            MPI_Win_unlock_all(shmem_sheap_win);

            MPI_Win_free(&shmem_etext_win);
            MPI_Win_free(&shmem_sheap_win);

#ifdef USE_SMP_OPTIMIZATIONS
            free(shmem_smp_rank_list);
            MPI_Group_free(&SHMEM_GROUP_NODE);
            MPI_Comm_free(&SHMEM_COMM_NODE);
#endif

            MPI_Group_free(&SHMEM_GROUP_WORLD);
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
int _num_pes(void) { return shmem_world_size; }
int shmem_n_pes(void) { return shmem_world_size; }
int _my_pe(void) { return shmem_world_rank; }
int shmem_my_pe(void) { return shmem_world_rank; }

/* return 0 on successful lookup, otherwise 1 */
static inline int __shmem_window_offset(const void *address, const int pe, /* IN  */
                                        enum shmem_window_id_e * win_id,   /* OUT */
                                        shmem_offset_t * win_offset)       /* OUT */
{
#if SHMEM_DEBUG>3
    printf("[%d] __shmem_window_offset: address=%p, pe=%d \n", shmem_world_rank, address, pe);
    fflush(stdout);
#endif

#if SHMEM_DEBUG>5
    printf("[%d] shmem_etext_mybase_ptr=%p \n", shmem_world_rank, shmem_etext_mybase_ptr );
    printf("[%d] shmem_sheap_mybase_ptr=%p \n", shmem_world_rank, shmem_sheap_mybase_ptr );
    fflush(stdout);
#endif

    ptrdiff_t etext_offset = address - shmem_etext_mybase_ptr;
    ptrdiff_t sheap_offset = address - shmem_sheap_mybase_ptr;

    if (0 <= etext_offset && etext_offset <= shmem_etext_size) { 
        *win_offset = etext_offset;
        *win_id     = SHMEM_ETEXT_WINDOW;
#if SHMEM_DEBUG>5
        printf("[%d] found address in etext window \n", shmem_world_rank);
        printf("[%d] win_offset=%ld \n", shmem_world_rank, *win_offset);
#endif
        return 0;
    } else 
    if (0 <= sheap_offset && sheap_offset <= shmem_sheap_size) { 
        *win_offset = sheap_offset;
        *win_id     = SHMEM_SHEAP_WINDOW;
#if SHMEM_DEBUG>5
        printf("[%d] found address in sheap window \n", shmem_world_rank);
        printf("[%d] win_offset=%ld \n", shmem_world_rank, *win_offset);
#endif
        return 0;
    }
    else {
        *win_offset  = (shmem_offset_t)NULL;
        *win_id      = SHMEM_INVALID_WINDOW;
#if SHMEM_DEBUG>5
        printf("[%d] did not find address in a valid window \n", shmem_world_rank);
#endif
        return 1;
    }
}

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
}

void *shmalloc(size_t size)
{
    /* use page size for debugging purposes */
    const int default_alignment = 4096;
    return shmemalign(default_alignment, size);
}

void *shrealloc(void *ptr, size_t size)
{
    __shmem_abort(size, "shrealloc is not implemented in the hack-tastic version of sheap");
    return NULL;
}
void shfree(void *ptr)
{
    __shmem_warn("shfree is a no-op in the hack-tastic version of sheap");
    return;
}

/* quiet and fence are all about ordering.  
 * If put is already ordered, then these are no-ops.
 * fence only works on a single (implicit) remote PE, so
 * we track the last one that was targeted.
 * If any remote PE has been targeted, then quiet 
 * will flush all PEs. 
 */

static inline void __shmem_remote_sync(void)
{
    MPI_Win_flush_all(shmem_sheap_win);
    MPI_Win_flush_all(shmem_etext_win);
}

static inline void __shmem_local_sync(void)
{
    MPI_Win_sync(shmem_sheap_win);
    MPI_Win_sync(shmem_etext_win);
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

static inline void __shmem_rma(enum shmem_rma_type_e rma, MPI_Datatype mpi_type,
                               void *target, const void *source, size_t len, int pe)
{
    enum shmem_window_id_e win_id;
    shmem_offset_t win_offset;
    MPI_Win win;

#if SHMEM_DEBUG>3
    printf("[%d] __shmem_rma: rma=%d, type=%d, target=%p, source=%p, len=%zu, pe=%d \n", 
            shmem_world_rank, rma, mpi_type, target, source, len, pe);
    fflush(stdout);
#endif

    int count = 0;
    if ( likely(len<(size_t)INT32_MAX) ) {
        count = len;
    } else {
        /* TODO generate derived type ala BigMPI */
        __shmem_abort(rma, "count exceeds the range of a 32b integer");
    }

    switch (rma) {
        case SHMEM_PUT:
            assert(0==__shmem_window_offset(target, pe, &win_id, &win_offset));
#if SHMEM_DEBUG>3
            printf("[%d] win_id=%d, offset=%lld \n", 
                   shmem_world_rank, win_id, (long long)win_offset);
            fflush(stdout);
#endif
            win = (win_id==SHMEM_ETEXT_WINDOW) ? shmem_etext_win : shmem_sheap_win;
#ifdef USE_SMP_OPTIMIZATIONS
            if (shmem_world_is_smp) {
            } else 
#endif
            {
#ifdef USE_ORDERED_RMA
                MPI_Accumulate(source, count, mpi_type,                   /* origin */
                               pe, (MPI_Aint)win_offset, count, mpi_type, /* target */
                               MPI_REPLACE,                               /* atomic, ordered Put */
                               win);
#else
                MPI_Put(source, count, mpi_type,                   /* origin */
                        pe, (MPI_Aint)win_offset, count, mpi_type, /* target */
                        win);
#endif
                MPI_Win_flush_local(pe, win);
            }
            break;
        case SHMEM_GET:
            assert(0==__shmem_window_offset(source, pe, &win_id, &win_offset));
#if SHMEM_DEBUG>3
            printf("[%d] win_id=%d, offset=%lld \n", 
                   shmem_world_rank, win_id, (long long)win_offset);
            fflush(stdout);
#endif
            win = (win_id==SHMEM_ETEXT_WINDOW) ? shmem_etext_win : shmem_sheap_win;
#ifdef USE_SMP_OPTIMIZATIONS
            if (shmem_world_is_smp) {
            } else 
#endif
            {
#ifdef USE_ORDERED_RMA
                MPI_Get_accumulate(NULL, 0, MPI_DATATYPE_NULL,                /* origin */
                                   target, count, mpi_type,                   /* result */
                                   pe, (MPI_Aint)win_offset, count, mpi_type, /* remote */
                                   MPI_NO_OP,                                 /* atomic, ordered Get */
                                   win);
#else
                MPI_Get(target, count, mpi_type,                   /* result */
                        pe, (MPI_Aint)win_offset, count, mpi_type, /* remote */
                        win);
#endif
                MPI_Win_flush(pe, win);
            }
            break;
#if 0
        case SHMEM_IPUT:
            assert(0==__shmem_window_offset(target, pe, &win_id, &win_offset));
            break;
        case SHMEM_IGET:
            assert(0==__shmem_window_offset(target, pe, &win_id, &win_offset));
            break;
#endif
        default:
            __shmem_abort(rma, "Unsupported RMA type.");
            break;
    }
    return;
}

/* 8.6: Elemental Put Routines */
void shmem_float_p(float *addr, float v, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_FLOAT, addr, &v, 1, pe);
}
void shmem_double_p(double *addr, double v, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_DOUBLE, addr, &v, 1, pe);
}
void shmem_longdouble_p(long double *addr, long double v, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_LONG_DOUBLE, addr, &v, 1, pe);
}
void shmem_char_p(char *addr, char v, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_CHAR, addr, &v, 1, pe);
}
void shmem_short_p(short *addr, short v, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_SHORT, addr, &v, 1, pe);
}
void shmem_int_p(int *addr, int v, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_INT, addr, &v, 1, pe);
}
void shmem_long_p(long *addr, long v, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_LONG, addr, &v, 1, pe);
}
void shmem_longlong_p(long long *addr, long long v, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_LONG_LONG, addr, &v, 1, pe);
}

/* 8.7: Block Data Put Routines */
void shmem_float_put(float *target, const float *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_FLOAT, target, source, len, pe);
}
void shmem_double_put(double *target, const double *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_DOUBLE, target, source, len, pe);
}
void shmem_longdouble_put(long double *target, const long double *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_LONG_DOUBLE, target, source, len, pe);
}
void shmem_char_put(char *target, const char *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_CHAR, target, source, len, pe);
}
void shmem_short_put(short *target, const short *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_SHORT, target, source, len, pe);
}
void shmem_int_put(int *target, const int *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_INT, target, source, len, pe);
}
void shmem_long_put(long *target, const long *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_LONG, target, source, len, pe);
}
void shmem_longlong_put(long long *target, const long long *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_LONG_LONG, target, source, len, pe);
}
void shmem_putmem(void *target, const void *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_BYTE, target, source, len, pe);
}
void shmem_put32(void *target, const void *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_INT32_T, target, source, len, pe);
}
void shmem_put64(void *target, const void *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_INT64_T, target, source, len, pe);
}
void shmem_put128(void *target, const void *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_INT64_T, target, source, 2*len, pe);
}
void shmem_complexf_put(float complex * target, const float complex * source, size_t len, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_COMPLEX, target, source, len, pe);
}
void shmem_complexd_put(double complex * target, const double complex * source, size_t len, int pe)
{
    __shmem_rma(SHMEM_PUT, MPI_DOUBLE_COMPLEX, target, source, len, pe);
}

/* 8.9: Elemental Data Get Routines */
float shmem_float_g(float *addr, int pe)
{
    float v;
    __shmem_rma(SHMEM_GET, MPI_FLOAT, &v, addr, 1, pe);
    return v;
}
double shmem_double_g(double *addr, int pe)
{
    double v;
    __shmem_rma(SHMEM_GET, MPI_DOUBLE, &v, addr, 1, pe);
    return v;
}
long double shmem_longdouble_g(long double *addr, int pe)
{
    long double v;
    __shmem_rma(SHMEM_GET, MPI_LONG_DOUBLE, &v, addr, 1, pe);
    return v;
}
char shmem_char_g(char *addr, int pe)
{
    char v;
    __shmem_rma(SHMEM_GET, MPI_CHAR, &v, addr, 1, pe);
    return v;
}
short shmem_short_g(short *addr, int pe)
{
    short v;
    __shmem_rma(SHMEM_GET, MPI_SHORT, &v, addr, 1, pe);
    return v;
}
int shmem_int_g(int *addr, int pe)
{
    int v;
    __shmem_rma(SHMEM_GET, MPI_INT, &v, addr, 1, pe);
    return v;
}
long shmem_long_g(long *addr, int pe)
{
    long v;
    __shmem_rma(SHMEM_GET, MPI_LONG, &v, addr, 1, pe);
    return v;
}
long long shmem_longlong_g(long long *addr, int pe)
{
    long long v;
    __shmem_rma(SHMEM_GET, MPI_LONG_LONG, &v, addr, 1, pe);
    return v;
}

/* 8.10 Block Data Get Routines */
void shmem_float_get(float *target, const float *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_GET, MPI_FLOAT, target, source, len, pe);
}
void shmem_double_get(double *target, const double *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_GET, MPI_DOUBLE, target, source, len, pe);
}
void shmem_longdouble_get(long double *target, const long double *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_GET, MPI_LONG_DOUBLE, target, source, len, pe);
}
void shmem_char_get(char *target, const char *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_GET, MPI_CHAR, target, source, len, pe);
}
void shmem_short_get(short *target, const short *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_GET, MPI_SHORT, target, source, len, pe);
}
void shmem_int_get(int *target, const int *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_GET, MPI_INT, target, source, len, pe);
}
void shmem_long_get(long *target, const long *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_GET, MPI_LONG, target, source, len, pe);
}
void shmem_longlong_get(long long *target, const long long *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_GET, MPI_LONG_LONG, target, source, len, pe);
}
void shmem_getmem(void *target, const void *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_GET, MPI_BYTE, target, source, len, pe);
}
void shmem_get32(void *target, const void *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_GET, MPI_INT32_T, target, source, len, pe);
}
void shmem_get64(void *target, const void *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_GET, MPI_INT64_T, target, source, len, pe);
}
void shmem_get128(void *target, const void *source, size_t len, int pe)
{
    __shmem_rma(SHMEM_GET, MPI_INT64_T, target, source, 2*len, pe);
}
void shmem_complexf_get(float complex * target, const float complex * source, size_t len, int pe)
{
    __shmem_rma(SHMEM_GET, MPI_COMPLEX, target, source, len, pe);
}
void shmem_complexd_get(double complex * target, const double complex * source, size_t len, int pe)
{
    __shmem_rma(SHMEM_GET, MPI_DOUBLE_COMPLEX, target, source, len, pe);
}

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

    assert(0==__shmem_window_offset(remote, pe, &win_id, &win_offset));
    MPI_Win win = (win_id==SHMEM_ETEXT_WINDOW) ? shmem_etext_win : shmem_sheap_win;

    switch (amo) {
        case SHMEM_SWAP:
#ifdef USE_SMP_OPTIMIZATIONS
            if (shmem_world_is_smp) {
            } else 
#endif
            {
                MPI_Fetch_and_op(input, output, mpi_type, pe, win_offset, MPI_REPLACE, win);
                MPI_Win_flush(pe, win);
            }
            break;
        case SHMEM_CSWAP:
#ifdef USE_SMP_OPTIMIZATIONS
            if (shmem_world_is_smp) {
            } else 
#endif
            {
                MPI_Compare_and_swap(input, compare, output, mpi_type, pe, win_offset, win);
                MPI_Win_flush(pe, win);
            }
            break;
        /* (F)INC = (F)ADD w/ input=1 at the higher level */
        case SHMEM_ADD:
#ifdef USE_SMP_OPTIMIZATIONS
            if (shmem_world_is_smp) {
            } else 
#endif
            {
                MPI_Fetch_and_op(input, NULL, mpi_type, pe, win_offset, MPI_SUM, win);
                MPI_Win_flush(pe, win);
            }
            break;
        case SHMEM_FADD:
#ifdef USE_SMP_OPTIMIZATIONS
            if (shmem_world_is_smp) {
            } else 
#endif
            {
                MPI_Fetch_and_op(input, output, mpi_type, pe, win_offset, MPI_SUM, win);
                MPI_Win_flush(pe, win);
            }
            break;
        default:
            __shmem_abort(amo, "Unsupported AMO type.");
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
float     shmem_float_swap(float *t, float v, int pe)            
{ 
    float     r; 
    __shmem_amo(SHMEM_SWAP, MPI_FLOAT,     &r, t, &v, NULL, pe) ; 
    return r; 
}
double    shmem_double_swap(double *t, double v, int pe)         
{ 
    double    r; 
    __shmem_amo(SHMEM_SWAP, MPI_DOUBLE,    &r, t, &v, NULL, pe) ; 
    return r; 
}
int       shmem_int_swap(int *t, int v, int pe)                  
{ 
    int       r; 
    __shmem_amo(SHMEM_SWAP, MPI_INT,       &r, t, &v, NULL, pe) ; 
    return r; 
}
long      shmem_long_swap(long *t, long v, int pe)               
{ 
    long      r; 
    __shmem_amo(SHMEM_SWAP, MPI_LONG,      &r, t, &v, NULL, pe) ; 
    return r; 
}
long long shmem_longlong_swap(long long *t, long long v, int pe) 
{ 
    long long r; 
    __shmem_amo(SHMEM_SWAP, MPI_LONG_LONG, &r, t, &v, NULL, pe) ; 
    return r; 
}
long      shmem_swap(long *t, long v, int pe)                    
{ 
    long      r; 
    __shmem_amo(SHMEM_SWAP, MPI_LONG,      &r, t, &v, NULL, pe) ; 
    return r; 
}

/* 8.12: Atomic Memory fetch-and-operate Routines -- Cswap */
int       shmem_int_cswap(int *t, int c, int v, int pe)                         
{ 
    int       r; 
    __shmem_amo(SHMEM_CSWAP, MPI_INT,       &r, t, &v, &c, pe) ; 
    return r; 
}
long      shmem_long_cswap(long *t, long c, long v, int pe)                     
{ 
    long      r; 
    __shmem_amo(SHMEM_CSWAP, MPI_LONG,      &r, t, &v, &c, pe) ; 
    return r; 
}
long long shmem_longlong_cswap(long long * t, long long c, long long v, int pe) 
{ 
    long long r; 
    __shmem_amo(SHMEM_CSWAP, MPI_LONG_LONG, &r, t, &v, &c, pe) ; 
    return r; 
}

/* 8.12: Atomic Memory fetch-and-operate Routines -- Fetch and Add */
int       shmem_int_fadd(int *t, int v, int pe)                  
{ 
    int       r; 
    __shmem_amo(SHMEM_FADD, MPI_INT,       &r, t, &v, NULL, pe); 
    return r; 
}
long      shmem_long_fadd(long *t, long v, int pe)               
{ 
    long      r; 
    __shmem_amo(SHMEM_FADD, MPI_LONG,      &r, t, &v, NULL, pe); 
    return r; 
}
long long shmem_longlong_fadd(long long *t, long long v, int pe) 
{ 
    long long r; 
    __shmem_amo(SHMEM_FADD, MPI_LONG_LONG, &r, t, &v, NULL, pe); 
    return r; 
}

/* 8.12: Atomic Memory fetch-and-operate Routines -- Fetch and Increment */
int       shmem_int_finc(int *t, int pe)             
{ 
    int v=1; 
    int r; 
    __shmem_amo(SHMEM_FADD, MPI_INT,       &r, t, &v, NULL, pe); 
    return r; 
}
long      shmem_long_finc(long *t, int pe)           
{ 
    long v=1; 
    long r; 
    __shmem_amo(SHMEM_FADD, MPI_LONG,      &r, t, &v, NULL, pe); 
    return r; 
}
long long shmem_longlong_finc(long long *t, int pe)  
{ 
    long long v=1; 
    long long r; 
    __shmem_amo(SHMEM_FADD, MPI_LONG_LONG, &r, t, &v, NULL, pe); 
    return r; 
}

/* 8.13: Atomic Memory Operation Routines -- Add */
void shmem_int_add(int *t, int v, int pe)                  
{ 
    __shmem_amo(SHMEM_ADD, MPI_INT,       NULL, t, &v, NULL, pe); 
}
void shmem_long_add(long *t, long v, int pe)               
{ 
    __shmem_amo(SHMEM_ADD, MPI_LONG,      NULL, t, &v, NULL, pe); 
}
void shmem_longlong_add(long long *t, long long v, int pe) 
{ 
    __shmem_amo(SHMEM_ADD, MPI_LONG_LONG, NULL, t, &v, NULL, pe); 
}

/* 8.13: Atomic Memory Operation Routines -- Increment */
void shmem_int_inc(int *t, int pe)            
{ 
    int       v=1; 
    __shmem_amo(SHMEM_ADD, MPI_INT,       NULL, t, &v, NULL, pe); 
}
void shmem_long_inc(long *t, int pe)          
{ 
    long      v=1; 
    __shmem_amo(SHMEM_ADD, MPI_LONG,      NULL, t, &v, NULL, pe); 
}
void shmem_longlong_inc(long long *t, int pe) 
{ 
    long long v=1; 
    __shmem_amo(SHMEM_ADD, MPI_LONG_LONG, NULL, t, &v, NULL, pe); 
}

#include "shmem-wait.h"

/* 8.14: Point-to-Point Synchronization Routines -- Wait */
void shmem_short_wait(short *var, short v) { SHMEM_WAIT(var,v); }
void shmem_int_wait(int *var, int v) { SHMEM_WAIT(var,v); }
void shmem_long_wait(long *var, long v) { SHMEM_WAIT(var,v); }
void shmem_longlong_wait(long long *var, long long v) { SHMEM_WAIT(var,v); }
void shmem_wait(long *var, long v) { SHMEM_WAIT(var,v); }

#if 0
/* 8.14: Point-to-Point Synchronization Routines -- Wait Until */
void shmem_short_wait_until(short *var, int c, short v)
void shmem_int_wait_until(int *var, int c, int v)
void shmem_long_wait_until(long *var, int c, long v)
void shmem_longlong_wait_until(long long *var, int c, long long v)
void shmem_wait_until(long *ivar, int cmp, long v)
#endif

/* TODO 
 * One might assume that the same subcomms are used more than once and thus caching these is prudent.
 */
static inline void __shmem_create_comm(int pe_start, int log_pe_stride, int pe_size,       /* IN  */
                                               MPI_Comm * comm, MPI_Group * strided_group) /* OUT */
{
    int * pe_list = malloc(pe_size*sizeof(int)); assert(pe_list);

    int pe_stride = 1<<log_pe_stride;
    for (int i=0; i<pe_size; i++)
        pe_list[i] = pe_start + i*pe_stride;

    MPI_Group_incl(SHMEM_GROUP_WORLD, pe_size, pe_list, strided_group);
    MPI_Comm_create_group(SHMEM_COMM_WORLD, *strided_group, 0 /* tag */, comm); /* collective on group */

    free(pe_list);

    return;
}

static inline void __shmem_coll(enum shmem_coll_type_e coll, MPI_Datatype mpi_type, MPI_Op reduce_op,
                                void * target, const void * source, size_t len, 
                                int pe_root, int pe_start, int log_pe_stride, int pe_size)
{
    int collective_on_world = (pe_start==0 && log_pe_stride==0 && pe_size==shmem_world_size);

    MPI_Comm  comm;
    MPI_Group strided_group;

    if (collective_on_world) {
        comm = SHMEM_COMM_WORLD;
    } else {
        /* TODO implement communicator cache */
        __shmem_create_comm(pe_start, log_pe_stride, pe_size, &comm, &strided_group);
    }

    int count = 0;
    if ( likely(len<(size_t)INT32_MAX) ) {
        count = len;
    } else {
        /* TODO generate derived type ala BigMPI */
        __shmem_abort(coll, "count exceeds the range of a 32b integer");
    }

    switch (coll) {
        case SHMEM_BARRIER:
            MPI_Barrier( comm );
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
                if (pe_root==shmem_world_rank) {
                    int typesize;
                    MPI_Type_size(mpi_type, &typesize); /* could optimize away since only two cases possible */
                    memcpy(target, source, count*typesize);
                }
                MPI_Bcast(target, count, mpi_type, bcast_root, comm); 
            }
            break;
        case SHMEM_ALLGATHER:
            MPI_Allgather(source, count, mpi_type, target, count, mpi_type, comm);
            break;
        case SHMEM_ALLGATHERV:
            {
                int * rcounts = malloc(pe_size*sizeof(int)); assert(rcounts!=NULL);
                int * rdispls = malloc(pe_size*sizeof(int)); assert(rdispls!=NULL);
                MPI_Allgather(&count, 1, MPI_INT, rcounts, 1, MPI_INT, comm);
                rdispls[0] = 0;
                for (int i=1; i<pe_size; i++) {
                    rdispls[i] = rdispls[i-1] + rcounts[i-1];
                }
                MPI_Allgatherv(source, count, mpi_type, target, rcounts, rdispls, mpi_type, comm);
                free(rdispls);
                free(rcounts);
            }
            break;
        case SHMEM_ALLREDUCE:
            /* TODO detect partially overlapping buffers */
            MPI_Allreduce((source==target) ? MPI_IN_PLACE : source, target, count, mpi_type, reduce_op, comm);
            break;
        default:
            __shmem_abort(coll, "Unsupported collective type.");
            break;
    }

    if (!collective_on_world) {
        MPI_Group_free(&strided_group);
        MPI_Comm_free(&comm);
    }
    return;
}

/* 8.15: Barrier Synchronization Routines */

void shmem_barrier(int PE_start, int logPE_stride, int PE_size, long *pSync)
{ 
    __shmem_remote_sync();
    __shmem_local_sync();
    __shmem_coll(SHMEM_BARRIER, MPI_DATATYPE_NULL, MPI_OP_NULL, NULL, NULL, 0 /* count */, 0 /* root */,  PE_start, logPE_stride, PE_size); 
}

void shmem_barrier_all(void) 
{ 
    __shmem_remote_sync();
    __shmem_local_sync();
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

void shmem_short_and_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_SHORT, MPI_LAND, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_int_and_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_INT, MPI_LAND, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_long_and_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG, MPI_LAND, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longlong_and_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_LONG, MPI_LAND, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}

void shmem_short_or_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_SHORT, MPI_BOR, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_int_or_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_INT, MPI_BOR, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_long_or_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG, MPI_BOR, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longlong_or_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_LONG, MPI_BOR, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}

void shmem_short_xor_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_SHORT, MPI_BXOR, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_int_xor_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_INT, MPI_BXOR, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_long_xor_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG, MPI_BXOR, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longlong_xor_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_LONG, MPI_BXOR, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}

void shmem_float_min_to_all(float *target, float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_FLOAT, MPI_MIN, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_double_min_to_all(double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_DOUBLE, MPI_MIN, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longdouble_min_to_all(long double *target, long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_DOUBLE, MPI_MIN, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_short_min_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_SHORT, MPI_MIN, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_int_min_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_INT, MPI_MIN, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_long_min_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG, MPI_MIN, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longlong_min_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_LONG, MPI_MIN, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}

void shmem_float_max_to_all(float *target, float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_FLOAT, MPI_MAX, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_double_max_to_all(double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_DOUBLE, MPI_MAX, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longdouble_max_to_all(long double *target, long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_DOUBLE, MPI_MAX, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_short_max_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_SHORT, MPI_MAX, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_int_max_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_INT, MPI_MAX, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_long_max_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG, MPI_MAX, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longlong_max_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_LONG, MPI_MAX, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}

void shmem_float_sum_to_all(float *target, float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_FLOAT, MPI_SUM, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_double_sum_to_all(double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_DOUBLE, MPI_SUM, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longdouble_sum_to_all(long double *target, long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_DOUBLE, MPI_SUM, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_short_sum_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_SHORT, MPI_SUM, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_int_sum_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_INT, MPI_SUM, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_long_sum_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG, MPI_SUM, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longlong_sum_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_LONG, MPI_SUM, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}

void shmem_float_prod_to_all(float *target, float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_FLOAT, MPI_PROD, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_double_prod_to_all(double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_DOUBLE, MPI_PROD, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longdouble_prod_to_all(long double *target, long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_DOUBLE, MPI_PROD, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_short_prod_to_all(short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_SHORT, MPI_PROD, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_int_prod_to_all(int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_INT, MPI_PROD, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_long_prod_to_all(long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG, MPI_PROD, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}
void shmem_longlong_prod_to_all(long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
    __shmem_coll(SHMEM_ALLREDUCE, MPI_LONG_LONG, MPI_PROD, target, source, nreduce, 0 /* root */, PE_start, logPE_stride, PE_size);
}

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
    /* In general, nodename != procname, of course, but there are 
     * many implementations where this will be true because the 
     * procname is just the IP address. */
    int namelen = 0;
    memset(shmem_procname, '\0', MPI_MAX_PROCESSOR_NAME);
    MPI_Get_processor_name( shmem_procname, &namelen );
    return shmem_procname;
}


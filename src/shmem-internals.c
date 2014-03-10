/* BSD-2 License.  Written by Jeff Hammond. */

#include "shmem-internals.h"

/* this code deals with SHMEM communication out of symmetric but non-heap data */
#if defined(__APPLE__)
#warning Global data support is not working yet.
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
    static unsigned long get_end()   { return &_end;   }
    static unsigned long get_etext() { return &_etext; }
    static unsigned long get_edata() { return &_edata; }
#elif defined(__linux__)
    /* http://man7.org/linux/man-pages/man3/end.3.html */
    extern char data_start;
    extern char etext;
    extern char edata;
    extern char end;
    static unsigned long get_sdata() { return (unsigned long)&data_start;   }
    static unsigned long get_end()   { return (unsigned long)&end;   }
    /* Static causes the compiler to warn that these are unused, which is correct. */
    //static unsigned long get_etext() { return (unsigned long)&etext; }
    //static unsigned long get_edata() { return (unsigned long)&edata; }
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
extern MPI_Comm  SHMEM_COMM_WORLD;
extern MPI_Group SHMEM_GROUP_WORLD; /* used for creating logpe comms */

extern int       shmem_is_initialized;
extern int       shmem_is_finalized;
extern int       shmem_world_size, shmem_world_rank;
extern char      shmem_procname[MPI_MAX_PROCESSOR_NAME];

#ifdef USE_SMP_OPTIMIZATIONS
extern MPI_Comm  SHMEM_COMM_NODE;
extern MPI_Group SHMEM_GROUP_NODE; /* may not be needed as global */
extern int       shmem_world_is_smp;
extern int       shmem_node_size, shmem_node_rank;
extern int *     shmem_smp_rank_list;
extern void **   shmem_smp_sheap_ptrs;
#endif

/* TODO probably want to make these 5 things into a struct typedef */
extern MPI_Win shmem_etext_win;
extern int     shmem_etext_size;
extern void *  shmem_etext_base_ptr;

extern MPI_Win shmem_sheap_win;
extern int     shmem_sheap_size;
extern void *  shmem_sheap_base_ptr;
extern void *  shmem_sheap_current_ptr;

#ifdef ENABLE_MPMD_SUPPORT
extern int     shmem_running_mpmd;
extern int     shmem_mpmd_my_appnum;
extern MPI_Win shmem_mpmd_appnum_win;
#endif

/*****************************************************************/

/* Reduce overhead of MPI_Type_size in MPI-bypass Put/Get path. */
#ifdef MPICH
#define FAST_Type_size(a) (((a)&0x0000ff00)>>8)
#define USE_MPICH_MACROS
#endif

/*****************************************************************/

void __shmem_warn(char * message)
{
#if SHMEM_DEBUG > 0
    printf("[%d] %s \n", shmem_world_rank, message);
    fflush(stdout);
#endif
    return;
}

void __shmem_abort(int code, char * message)
{
    printf("[%d] %s \n", shmem_world_rank, message);
    fflush(stdout);
    MPI_Abort(SHMEM_COMM_WORLD, code);
    return;
}

#if 0
/* This function is not used because we do not need this information. */
int __shmem_address_is_symmetric(size_t my_sheap_base_ptr)
{
    /* I am not sure if there is a better way to operate on addresses... */
    /* cannot fuse allreduces because max{base,-base} trick does not work for unsigned */

    int is_symmetric = 0;
    size_t minbase = 0;
    size_t maxbase = 0;

    /* The latter might be faster on machines with bad collective implementations. 
     * On Blue Gene, Allreduce is definitely the way to go. 
     */
    MPI_Reduce( &my_sheap_base_ptr, &minbase, 1, sizeof(size_t)==4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG, MPI_MIN, 0, SHMEM_COMM_WORLD );
    MPI_Reduce( &my_sheap_base_ptr, &maxbase, 1, sizeof(size_t)==4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG, MPI_MAX, 0, SHMEM_COMM_WORLD );
    if (shmem_world_rank==0)
        is_symmetric = ((minbase==my_sheap_base_ptr && my_sheap_base_ptr==maxbase) ? 1 : 0);
    MPI_Bcast( &is_symmetric, 1, MPI_INT, 0, SHMEM_COMM_WORLD );
    return is_symmetric;
}
#endif

void __shmem_initialize(int threading)
{
    {
        int flag;
        MPI_Initialized(&flag);

        int provided;
        if (!flag) {
            MPI_Init_thread(NULL, NULL, threading, &provided);
        } else {
            MPI_Query_thread(&provided);
        }

        if (threading<provided)
            __shmem_abort(provided, "Your MPI implementation did not provide the requested thread support.");
    }

    if (!shmem_is_initialized) {

        MPI_Comm_dup(MPI_COMM_WORLD, &SHMEM_COMM_WORLD);
        MPI_Comm_size(SHMEM_COMM_WORLD, &shmem_world_size);
        MPI_Comm_rank(SHMEM_COMM_WORLD, &shmem_world_rank);
        MPI_Comm_group(SHMEM_COMM_WORLD, &SHMEM_GROUP_WORLD);

        {
            /* Check for MPMD usage. */
            int appnum=0;
            int is_set=0;
            MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_APPNUM, &appnum, &is_set);
#ifndef ENABLE_MPMD_SUPPORT
            /* If any rank detects MPMD, we abort.  No need to check collectively. */
            if (is_set && appnum)
                __shmem_abort(appnum, "You need to enable MPMD support in the build.");
#else
            /* This may not be necessary but it is safer to check on all ranks. */
            MPI_Allreduce(MPI_IN_PLACE, &is_set, 1, MPI_INT, MPI_MAX, SHMEM_COMM_WORLD);
            if (is_set) {
                shmem_mpmd_my_appnum = appnum;

                /* Check for appnum homogeneity. */
                int appmin, appmax;
                MPI_Allreduce(&appnum, &appmin, 1, MPI_INT, MPI_MIN, SHMEM_COMM_WORLD);
                MPI_Allreduce(&appnum, &appmax, 1, MPI_INT, MPI_MAX, SHMEM_COMM_WORLD);
                shmem_running_mpmd = (appmin != appmax) ? 1 : 0;
            } else {
                shmem_running_mpmd = 0;
            }

            if (shmem_running_mpmd) {
                /* Never going to need direct access; in any case, base is a window attribute. */
                void * shmem_mpmd_appnum_base;
                MPI_Win_allocate((MPI_Aint)sizeof(int), sizeof(int) /* disp_unit */, MPI_INFO_NULL, SHMEM_COMM_WORLD,
                                 &shmem_mpmd_appnum_base, &shmem_mpmd_appnum_win);
                MPI_Win_lock_all(0, shmem_mpmd_appnum_win);

                /* Write my appnum into appropriate location in window. */
                int junk;
                MPI_Fetch_and_op(&shmem_mpmd_my_appnum, &junk, MPI_INT, shmem_world_rank, 0,
                                 MPI_REPLACE, shmem_mpmd_appnum_win);
                MPI_Win_flush(shmem_world_rank, shmem_mpmd_appnum_win);
            }
#endif
        }

        if (shmem_world_rank==0) {
            char * c = getenv("SHMEM_SYMMETRIC_HEAP_SIZE");
            shmem_sheap_size = ( (c) ? atoi(c) : 128*1024*1024 );
        }
        MPI_Bcast( &shmem_sheap_size, 1, MPI_INT, 0, SHMEM_COMM_WORLD );

        MPI_Info sheap_info=MPI_INFO_NULL, etext_info=MPI_INFO_NULL;
        MPI_Info_create(&sheap_info);
        MPI_Info_create(&etext_info);

        /* We define the sheap size to be symmetric and assume it for the global static data. */
        MPI_Info_set(sheap_info, "same_size", "true");
        MPI_Info_set(etext_info, "same_size", "true");

        /* shmem_{put,get,swap,cswap} only requires REPLACE (atomic Put) and NO_OP (atomic Get). */
        /* We cannot use MPI_SUM if we enable this, hence the {inc,add,finc,fadd} atomics
         * are disabled by this option. */
#ifdef USE_SAME_OP_NO_OP
        MPI_Info_set(sheap_info, "accumulate_ops", "same_op_no_op");
        MPI_Info_set(etext_info, "accumulate_ops", "same_op_no_op");
#endif

#if ENABLE_RMA_ORDERING
        /* Given the additional synchronization overhead required,
         * there is no discernible performance benefit to this. */
        MPI_Info_set(sheap_info, "accumulate_ordering", "");
        MPI_Info_set(etext_info, "accumulate_ordering", "");
#endif

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

        if (shmem_world_is_smp) {
            /* There is no performance advantage associated with a contiguous layout of shared memory. */
            MPI_Info_set(sheap_info, "alloc_shared_noncontig", "true");

            MPI_Win_allocate_shared((MPI_Aint)shmem_sheap_size, 1 /* disp_unit */, sheap_info, SHMEM_COMM_WORLD, 
                                    &shmem_sheap_base_ptr, &shmem_sheap_win);
            shmem_smp_sheap_ptrs = malloc( shmem_node_size * sizeof(void*) ); assert(shmem_smp_sheap_ptrs!=NULL);
            for (int rank=0; rank<shmem_node_size; rank++) {
                MPI_Aint size; /* unused */
                int      disp; /* unused */
                MPI_Win_shared_query(shmem_sheap_win, rank, &size, &disp, &shmem_smp_sheap_ptrs[rank]);
            }
        } else 
#endif
        {
            MPI_Info_set(sheap_info, "alloc_shm", "true");
            MPI_Win_allocate((MPI_Aint)shmem_sheap_size, 1 /* disp_unit */, sheap_info, SHMEM_COMM_WORLD, 
                             &shmem_sheap_base_ptr, &shmem_sheap_win);
        }
        MPI_Win_lock_all(0, shmem_sheap_win);
        /* this is the hack-tastic sheap initialization */
        shmem_sheap_current_ptr = shmem_sheap_base_ptr;
#if SHMEM_DEBUG > 1
        printf("[%d] shmem_sheap_current_ptr  = %p  \n", shmem_world_rank, shmem_sheap_current_ptr );
        fflush(stdout);
#endif

        /* FIXME eliminate platform-specific stuff here i.e. find a way to move to top */
#if defined(__APPLE__)
	shmem_etext_base_ptr = (void*) get_etext();
#else
	shmem_etext_base_ptr = (void*) get_sdata();
#endif
        unsigned long long_etext_size   = get_end() - (unsigned long)shmem_etext_base_ptr;
        assert(long_etext_size<(unsigned long)INT32_MAX); 
        shmem_etext_size = (int)long_etext_size;

#if defined(__APPLE__) && SHMEM_DEBUG > 5
        printf("[%d] get_etext()       = %p \n", shmem_world_rank, (void*)get_etext() );
        printf("[%d] get_edata()       = %p \n", shmem_world_rank, (void*)get_edata() );
        printf("[%d] get_end()         = %p \n", shmem_world_rank, (void*)get_end()   );
        //printf("[%d] long_etext_size   = %lu \n", shmem_world_rank, long_etext_size );
        printf("[%d] shmem_etext_size  = %d  \n", shmem_world_rank, shmem_etext_size );
        //printf("[%d] my_etext_base_ptr = %p  \n", shmem_world_rank, my_etext_base_ptr );
        fflush(stdout);
#endif

#ifdef ABUSE_MPICH_FOR_GLOBALS
        MPI_Win_create_dynamic(etext_info, SHMEM_COMM_WORLD, &shmem_etext_win);
#else
        MPI_Win_create(shmem_etext_base_ptr, shmem_etext_size, 1 /* disp_unit */, etext_info, SHMEM_COMM_WORLD, 
                       &shmem_etext_win);
#endif
        MPI_Win_lock_all(0, shmem_etext_win);

        MPI_Info_free(&etext_info);
        MPI_Info_free(&sheap_info);

        /* It is hard if not impossible to implement SHMEM without the UNIFIED model. */
        {
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

        /* allocate Mutex */
	MCS_Mutex_create(shmem_world_size-1, SHMEM_COMM_WORLD, &hdl);

#if ENABLE_COMM_CACHING
        shmem_comm_cache_size = 16;
        comm_cache = malloc(shmem_comm_cache_size * sizeof(shmem_comm_t) ); assert(comm_cache!=NULL);
        for (int i=0; i<shmem_comm_cache_size; i++) {
            comm_cache[i].start = -1;
            comm_cache[i].logs  = -1;
            comm_cache[i].size  = -1;
            comm_cache[i].comm  = MPI_COMM_NULL;
            comm_cache[i].group = MPI_GROUP_NULL;
        }
#endif

        MPI_Barrier(SHMEM_COMM_WORLD);

        shmem_is_initialized = 1;
    }
    return;
}

void __shmem_finalize(void)
{
    int flag;
    MPI_Finalized(&flag);

    if (!flag) {
        if (shmem_is_initialized && !shmem_is_finalized) {

       	/* clear locking window */
	MCS_Mutex_free(&hdl);
#if ENABLE_COMM_CACHING
            for (int i=0; i<shmem_comm_cache_size; i++) {
                if (comm_cache[i].comm != MPI_COMM_NULL) {
                    MPI_Comm_free( &(comm_cache[i].comm) );
                    MPI_Group_free( &(comm_cache[i].group) );
                }
            }
            free(comm_cache);
#endif
            MPI_Barrier(SHMEM_COMM_WORLD);

#ifdef ENABLE_MPMD_SUPPORT
            if (shmem_running_mpmd) {
                MPI_Win_unlock_all(shmem_mpmd_appnum_win);
                MPI_Win_free(&shmem_mpmd_appnum_win);
            }
#endif
            MPI_Win_unlock_all(shmem_etext_win);
            MPI_Win_free(&shmem_etext_win);

            MPI_Win_unlock_all(shmem_sheap_win);
            MPI_Win_free(&shmem_sheap_win);

#ifdef USE_SMP_OPTIMIZATIONS
            if (shmem_world_is_smp)
                free(shmem_smp_sheap_ptrs);
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

/* quiet and fence are all about ordering.  
 * If put is already ordered, then these are no-ops.
 * fence only works on a single (implicit) remote PE, so
 * we track the last one that was targeted.
 * If any remote PE has been targeted, then quiet 
 * will flush all PEs. 
 */

void __shmem_remote_sync(int remote_completion)
{
#if ENABLE_RMA_ORDERING
    if (remote_completion)
#endif
    {
        MPI_Win_flush_all(shmem_sheap_win);
        MPI_Win_flush_all(shmem_etext_win);
    }
}

void __shmem_local_sync(void)
{
#ifdef USE_SMP_OPTIMIZATIONS
    __sync_synchronize();
#endif
    MPI_Win_sync(shmem_sheap_win);
    MPI_Win_sync(shmem_etext_win);
}

/* return 0 on successful lookup, otherwise 1 */
int __shmem_window_offset(const void *address, const int pe, /* IN  */
                          enum shmem_window_id_e * win_id,   /* OUT */
                          shmem_offset_t * win_offset)       /* OUT */
{
#if SHMEM_DEBUG>3
    printf("[%d] __shmem_window_offset: address=%p, pe=%d \n", shmem_world_rank, address, pe);
    fflush(stdout);
#endif

#if SHMEM_DEBUG>5
    printf("[%d] shmem_etext_base_ptr=%p \n", shmem_world_rank, shmem_etext_base_ptr );
    printf("[%d] shmem_sheap_base_ptr=%p \n", shmem_world_rank, shmem_sheap_base_ptr );
    fflush(stdout);
#endif

    ptrdiff_t sheap_offset = address - shmem_sheap_base_ptr;
    ptrdiff_t etext_offset = address - shmem_etext_base_ptr;

    if (0 <= sheap_offset && sheap_offset <= shmem_sheap_size) {
        *win_offset = sheap_offset;
        *win_id     = SHMEM_SHEAP_WINDOW;
#if SHMEM_DEBUG>5
        printf("[%d] found address in sheap window \n", shmem_world_rank);
        printf("[%d] win_offset=%ld \n", shmem_world_rank, *win_offset);
#endif
        return 0;
    }
    else if (0 <= etext_offset && etext_offset <= shmem_etext_size) {
        *win_offset = etext_offset;
        *win_id     = SHMEM_ETEXT_WINDOW;
#if SHMEM_DEBUG>5
        printf("[%d] found address in etext window \n", shmem_world_rank);
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

void __shmem_put(MPI_Datatype mpi_type, void *target, const void *source, size_t len, int pe)
{
    enum shmem_window_id_e win_id;
    shmem_offset_t win_offset;

#if SHMEM_DEBUG>3
    printf("[%d] __shmem_put: type=%d, target=%p, source=%p, len=%zu, pe=%d \n",
            shmem_world_rank, mpi_type, target, source, len, pe);
    fflush(stdout);
#endif

    int count = 0;
    if ( likely(len<(size_t)INT32_MAX) ) { /* need second check if size_t is signed */
        count = len;
    } else {
        /* TODO generate derived type ala BigMPI */
        __shmem_abort(len%INT32_MAX, "__shmem_put: count exceeds the range of a 32b integer");
    }

    if (__shmem_window_offset(target, pe, &win_id, &win_offset)) {
        __shmem_abort(pe, "__shmem_window_offset failed to find target");
    }

#if SHMEM_DEBUG>3
    printf("[%d] win_id=%d, offset=%lld \n", 
           shmem_world_rank, win_id, (long long)win_offset);
    fflush(stdout);
#endif

    MPI_Win win = (win_id==SHMEM_SHEAP_WINDOW) ? shmem_sheap_win : shmem_etext_win;

#ifdef USE_SMP_OPTIMIZATIONS
    if (shmem_world_is_smp && win_id==SHMEM_SHEAP_WINDOW) {
#ifdef USE_MPICH_MACROS
        int type_size = FAST_Type_size(mpi_type);
#else
        int type_size;
        MPI_Type_size(mpi_type, &type_size);
#endif
        void * ptr = shmem_smp_sheap_ptrs[pe] + (target - shmem_sheap_base_ptr);
        memcpy(ptr, source, len*type_size);
    } else 
#endif
    {
#if ENABLE_RMA_ORDERING
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
    return;
}

void __shmem_get(MPI_Datatype mpi_type, void *target, const void *source, size_t len, int pe)
{
    enum shmem_window_id_e win_id;
    shmem_offset_t win_offset;

#if SHMEM_DEBUG>3
    printf("[%d] __shmem_get: type=%d, target=%p, source=%p, len=%zu, pe=%d \n", 
            shmem_world_rank, mpi_type, target, source, len, pe);
    fflush(stdout);
#endif

    int count = 0;
    if ( likely(len<(size_t)INT32_MAX) ) { /* need second check if size_t is signed */
        count = len;
    } else {
        /* TODO generate derived type ala BigMPI */
        __shmem_abort(len%INT32_MAX, "__shmem_get: count exceeds the range of a 32b integer");
    }

    if (__shmem_window_offset(source, pe, &win_id, &win_offset)) {
        __shmem_abort(pe, "__shmem_window_offset failed to find source");
    }

#if SHMEM_DEBUG>3
    printf("[%d] win_id=%d, offset=%lld \n", 
           shmem_world_rank, win_id, (long long)win_offset);
    fflush(stdout);
#endif

    MPI_Win win = (win_id==SHMEM_SHEAP_WINDOW) ? shmem_sheap_win : shmem_etext_win;
#ifdef USE_SMP_OPTIMIZATIONS
    if (shmem_world_is_smp && win_id==SHMEM_SHEAP_WINDOW) {
#ifdef USE_MPICH_MACROS
        int type_size = FAST_Type_size(mpi_type);
#else
        int type_size;
        MPI_Type_size(mpi_type, &type_size);
#endif
        void * ptr = shmem_smp_sheap_ptrs[pe] + (source - shmem_sheap_base_ptr);
        memcpy(target, ptr, len*type_size);
    } else 
#endif
    {
#if ENABLE_RMA_ORDERING
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
        MPI_Win_flush_local(pe, win);
    }
    return;
}

void __shmem_put_strided(MPI_Datatype mpi_type, void *target, const void *source, 
                         ptrdiff_t target_ptrdiff, ptrdiff_t source_ptrdiff, size_t len, int pe)
{
#if SHMEM_DEBUG>3
    printf("[%d] __shmem_put_strided: type=%d, target=%p, source=%p, len=%zu, pe=%d \n", 
                    shmem_world_rank, mpi_type, target, source, len, pe);
    fflush(stdout);
#endif

    int count = 0;
    if ( likely(len<(size_t)INT32_MAX) ) { /* need second check if size_t is signed */
        count = len;
    } else {
        /* TODO generate derived type ala BigMPI */
        __shmem_abort(len%INT32_MAX, "__shmem_get: count exceeds the range of a 32b integer");
    }

    enum shmem_window_id_e win_id;
    shmem_offset_t win_offset;

    if (__shmem_window_offset(target, pe, &win_id, &win_offset)) {
        __shmem_abort(pe, "__shmem_window_offset failed to find source");
    }
#if SHMEM_DEBUG>3
    printf("[%d] win_id=%d, offset=%lld \n", 
           shmem_world_rank, win_id, (long long)win_offset);
    fflush(stdout);
#endif

    MPI_Win win = (win_id==SHMEM_SHEAP_WINDOW) ? shmem_sheap_win : shmem_etext_win;
#ifdef USE_SMP_OPTIMIZATIONS
    if (0) {
        /* TODO */
    } else 
#endif
    {
        assert( (ptrdiff_t)INT32_MIN<target_ptrdiff && target_ptrdiff<(ptrdiff_t)INT32_MAX );
        assert( (ptrdiff_t)INT32_MIN<source_ptrdiff && source_ptrdiff<(ptrdiff_t)INT32_MAX );

        int target_stride = (int) target_ptrdiff;
        int source_stride = (int) source_ptrdiff;

        MPI_Datatype source_type;
        MPI_Type_vector(count, 1, source_stride, mpi_type, &source_type);
        MPI_Type_commit(&source_type);

        MPI_Datatype target_type;
        if (target_stride!=source_stride) {
            MPI_Type_vector(count, 1, target_stride, mpi_type, &target_type);
            MPI_Type_commit(&target_type);
        } else {
            target_type = source_type;
        }

#if ENABLE_RMA_ORDERING
        MPI_Accumulate(source, 1, source_type,                   /* origin */
                       pe, (MPI_Aint)win_offset, 1, target_type, /* target */
                       MPI_REPLACE,                              /* atomic, ordered Put */
                       win);
#else
        MPI_Put(source, 1, source_type,                   /* origin */
                pe, (MPI_Aint)win_offset, 1, target_type, /* target */
                win);
#endif
        MPI_Win_flush_local(pe, win);

        if (target_stride!=source_stride) {
            MPI_Type_free(&target_type);
        }
        MPI_Type_free(&source_type);
    }

    return;
}

void __shmem_get_strided(MPI_Datatype mpi_type, void *target, const void *source, 
                         ptrdiff_t target_ptrdiff, ptrdiff_t source_ptrdiff, size_t len, int pe)
{
#if SHMEM_DEBUG>3
    printf("[%d] __shmem_get_strided: type=%d, target=%p, source=%p, len=%zu, pe=%d \n", 
                    shmem_world_rank, mpi_type, target, source, len, pe);
    fflush(stdout);
#endif

    int count = 0;
    if ( likely(len<(size_t)INT32_MAX) ) { /* need second check if size_t is signed */
        count = len;
    } else {
        /* TODO generate derived type ala BigMPI */
        __shmem_abort(len%INT32_MAX, "__shmem_get: count exceeds the range of a 32b integer");
    }

    enum shmem_window_id_e win_id;
    shmem_offset_t win_offset;

    if (__shmem_window_offset(source, pe, &win_id, &win_offset)) {
        __shmem_abort(pe, "__shmem_window_offset failed to find source");
    }
#if SHMEM_DEBUG>3
    printf("[%d] win_id=%d, offset=%lld \n", 
           shmem_world_rank, win_id, (long long)win_offset);
    fflush(stdout);
#endif

    MPI_Win win = (win_id==SHMEM_SHEAP_WINDOW) ? shmem_sheap_win : shmem_etext_win;
#ifdef USE_SMP_OPTIMIZATIONS
    if (0) {
        /* TODO */
    } else 
#endif
    {
        assert( (ptrdiff_t)INT32_MIN<target_ptrdiff && target_ptrdiff<(ptrdiff_t)INT32_MAX );
        assert( (ptrdiff_t)INT32_MIN<source_ptrdiff && source_ptrdiff<(ptrdiff_t)INT32_MAX );

        int target_stride = (int) target_ptrdiff;
        int source_stride = (int) source_ptrdiff;

        MPI_Datatype source_type;
        MPI_Type_vector(count, 1, source_stride, mpi_type, &source_type);
        MPI_Type_commit(&source_type);

        MPI_Datatype target_type;
        if (target_stride!=source_stride) {
            MPI_Type_vector(count, 1, target_stride, mpi_type, &target_type);
            MPI_Type_commit(&target_type);
        } else {
            target_type = source_type;
        }

#if ENABLE_RMA_ORDERING
        MPI_Get_accumulate(NULL, 0, MPI_DATATYPE_NULL,                   /* origin */
                           target, 1, target_type,                   /* result */
                           pe, (MPI_Aint)win_offset, 1, source_type, /* remote */
                           MPI_NO_OP,                                    /* atomic, ordered Get */
                           win);
#else
        MPI_Get(target, 1, target_type,                   /* result */
                pe, (MPI_Aint)win_offset, 1, source_type, /* remote */
                win);
#endif
        MPI_Win_flush_local(pe, win);

        if (target_stride!=source_stride) 
            MPI_Type_free(&target_type);
        MPI_Type_free(&source_type);
    }

    return;
}

void __shmem_swap(MPI_Datatype mpi_type, void *output, void *remote, const void *input, int pe)
{
    enum shmem_window_id_e win_id;
    shmem_offset_t win_offset;

    if (__shmem_window_offset(remote, pe, &win_id, &win_offset)) {
        __shmem_abort(pe, "__shmem_window_offset failed to find source");
    }

    MPI_Win win = (win_id==SHMEM_SHEAP_WINDOW) ? shmem_sheap_win : shmem_etext_win;

#ifdef USE_SMP_OPTIMIZATIONS
    if (0) {
    } else 
#endif
    {
        MPI_Fetch_and_op(input, output, mpi_type, pe, win_offset, MPI_REPLACE, win);
        MPI_Win_flush(pe, win);
    }
    return;
}

void __shmem_cswap(MPI_Datatype mpi_type, void *output, void *remote, const void *input, const void *compare, int pe)
{
    enum shmem_window_id_e win_id;
    shmem_offset_t win_offset;

    if (__shmem_window_offset(remote, pe, &win_id, &win_offset)) {
        __shmem_abort(pe, "__shmem_window_offset failed to find source");
    }

    MPI_Win win = (win_id==SHMEM_SHEAP_WINDOW) ? shmem_sheap_win : shmem_etext_win;

#ifdef USE_SMP_OPTIMIZATIONS
    if (0) {
    } else 
#endif
    {
        MPI_Compare_and_swap(input, compare, output, mpi_type, pe, win_offset, win);
        MPI_Win_flush(pe, win);
    }
    return;
}

#ifndef USE_SAME_OP_NO_OP

void __shmem_add(MPI_Datatype mpi_type, void *remote, const void *input, int pe)
{
    enum shmem_window_id_e win_id;
    shmem_offset_t win_offset;

    if (__shmem_window_offset(remote, pe, &win_id, &win_offset)) {
        __shmem_abort(pe, "__shmem_window_offset failed to find source");
    }

    MPI_Win win = (win_id==SHMEM_SHEAP_WINDOW) ? shmem_sheap_win : shmem_etext_win;

#ifdef USE_SMP_OPTIMIZATIONS
    if (0) {
    } else 
#endif
    {
        MPI_Accumulate(input, 1, mpi_type, pe, win_offset, 1, mpi_type, MPI_SUM, win);
        MPI_Win_flush_local(pe, win);
    }
    return;
}

void __shmem_fadd(MPI_Datatype mpi_type, void *output, void *remote, const void *input, int pe)
{
    enum shmem_window_id_e win_id;
    shmem_offset_t win_offset;

    if (__shmem_window_offset(remote, pe, &win_id, &win_offset)) {
        __shmem_abort(pe, "__shmem_window_offset failed to find source");
    }

    MPI_Win win = (win_id==SHMEM_SHEAP_WINDOW) ? shmem_sheap_win : shmem_etext_win;

#ifdef USE_SMP_OPTIMIZATIONS
    if (0) {
    } else 
#endif
    {
        MPI_Fetch_and_op(input, output, mpi_type, pe, win_offset, MPI_SUM, win);
        MPI_Win_flush(pe, win);
    }
    return;
}

#endif // USE_SAME_OP_NO_OP

static inline int __shmem_translate_root(MPI_Group strided_group, int pe_root)
{
#if SHMEM_DEBUG > 4
    printf("[%d] __shmem_translate_root(..,%d) \n", shmem_world_rank, pe_root);
#endif
    /* Broadcasts require us to translate the root from the world reference frame
     * to the strided subcommunicator frame. */
    {
        /* TODO
         * It should be possible to sidestep the generic translation for the 
         * special cases allowed by SHMEM. */
        int world_ranks[1] = { pe_root };
        int strided_ranks[1];
        MPI_Group_translate_ranks(SHMEM_GROUP_WORLD, 1 /* count */, world_ranks, 
                                  strided_group, strided_ranks);
        return strided_ranks[0];
    }
}

/* TODO 
 * One might assume that the same subcomms are used more than once and thus caching these is prudent.
 */
static inline void __shmem_acquire_comm(int pe_start, int pe_logs, int pe_size, /* IN  */ 
                                        MPI_Comm * comm,                        /* OUT */
                                        int pe_root,                            /* IN  */
                                        int * broot)                            /* OUT */
{
    /* fastpath for world */
    if (pe_start==0 && pe_logs==0 && pe_size==shmem_world_size) {
        *comm  = SHMEM_COMM_WORLD;
        *broot = pe_root;
        return;
    }

#if ENABLE_COMM_CACHING
    for (int i=0; i<shmem_comm_cache_size; i++)
    {
        if (pe_start == comm_cache[i].start &&
            pe_logs  == comm_cache[i].logs  &&
            pe_size  == comm_cache[i].size  ) 
        {
            *comm  = comm_cache[i].comm;
            if (pe_root>=0) {
                *broot = __shmem_translate_root(comm_cache[i].group, pe_root);
            }
            return;
        }
    }
#endif
    {
        MPI_Group strided_group;

        /* List of processes in the group that will be created. */
        int * pe_list = malloc(pe_size*sizeof(int)); assert(pe_list!=NULL);

        /* Implement 2^pe_logs with bitshift. */
        int pe_stride = 1<<pe_logs;
        for (int i=0; i<pe_size; i++)
            pe_list[i] = pe_start + i*pe_stride;

        MPI_Group_incl(SHMEM_GROUP_WORLD, pe_size, pe_list, &strided_group);
        /* Unlike the MPI-2 variant (MPI_Comm_create), this is only collective on the group. */
        /* We use pe_start as the tag because that should sufficiently disambiguate 
         * simultaneous calls to this function on disjoint groups. */
        MPI_Comm_create_group(SHMEM_COMM_WORLD, strided_group, pe_start /* tag */, comm); 

        if (pe_root>=0) {
            *broot = __shmem_translate_root(strided_group, pe_root);
        }

#if ENABLE_COMM_CACHING
        for (int i=0; i<shmem_comm_cache_size; i++) {
            if (comm_cache[i].comm == MPI_COMM_NULL ) {
                comm_cache[i].start = pe_start;
                comm_cache[i].logs  = pe_logs;
                comm_cache[i].size  = pe_size;
                comm_cache[i].comm  = *comm;
                comm_cache[i].group = strided_group;
                return;
            }
        }
#endif
        /* This point is reached only if caching fails so free the group here. */
        MPI_Group_free(&strided_group);

        free(pe_list);
    }
    return;
}

static inline void __shmem_release_comm(int pe_start, int pe_logs, int pe_size, /* IN  */ 
                                        MPI_Comm * comm)                        /* OUT */
{
    if (pe_start==0 && pe_logs==0 && pe_size==shmem_world_size) {
        return;
    }

#if ENABLE_COMM_CACHING
    for (int i=0; i<shmem_comm_cache_size; i++) {
        if (comm_cache[i].comm == *comm ) {
            /* If our comm is cached, do nothing. */
            return;
        }
    }
#endif
    {
        MPI_Comm_free(comm);
    }
    return;
}

void __shmem_coll(enum shmem_coll_type_e coll, MPI_Datatype mpi_type, MPI_Op reduce_op,
                  void * target, const void * source, size_t len, 
                  int pe_root, int pe_start, int pe_logs, int pe_size)
{
    int broot = 0;
    MPI_Comm comm;

    __shmem_acquire_comm(pe_start, pe_logs, pe_size, &comm, 
                         pe_root, &broot);

    int count = 0;
    if ( likely(len<(size_t)INT32_MAX) ) {
        count = len;
    } else {
        /* TODO 
         * Generate derived type ala BigMPI. */
        __shmem_abort(coll, "count exceeds the range of a 32b integer");
    }

    switch (coll) {
        case SHMEM_BARRIER:
            MPI_Barrier( comm );
            break;
        case SHMEM_BROADCAST:
            {
                /* For bcast, MPI uses one buffer but SHMEM uses two. */
                /* From the OpenSHMEM 1.0 specification:
                 * "The data is not copied to the target address on the PE specified by PE_root." */
                MPI_Bcast(shmem_world_rank==pe_root ? (void*) source : target, 
                         count, mpi_type, broot, comm); 
	    }
            break;
        case SHMEM_FCOLLECT:
            MPI_Allgather(source, count, mpi_type, target, count, mpi_type, comm);
            break;
        case SHMEM_COLLECT:
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
            /* From the OpenSHMEM 1.0 specification:
            "[The] source and target may be the same array, but they must not be overlapping arrays." */
            MPI_Allreduce((source==target) ? MPI_IN_PLACE : source, target, count, mpi_type, reduce_op, comm);
            break;
        default:
            __shmem_abort(coll, "Unsupported collective type.");
            break;
    }

    __shmem_release_comm(pe_start, pe_logs, pe_size, &comm);

    return;
}


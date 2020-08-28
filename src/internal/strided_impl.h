/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_STRIDED_IMPL_H
#define INTERNAL_STRIDED_IMPL_H

#include "oshmpi_impl.h"

typedef struct OSHMPI_dtype_cache_obj {
    size_t nelems;
    ptrdiff_t stride;
    MPI_Datatype dtype;
    size_t ext_nelems;
    MPI_Datatype sdtype;
    struct OSHMPI_dtype_cache_obj *next;
} OSHMPI_dtype_cache_obj_t;

typedef struct OSHMPI_dtype_cache_list {
    OSHMPI_dtype_cache_obj_t *head;
    int nobjs;
    OSHMPIU_thread_cs_t cs;
} OSHMPI_dtype_cache_t;

extern OSHMPI_dtype_cache_t OSHMPI_strided_dtype_cache;

#ifdef OSHMPI_ENABLE_STRIDED_DTYPE_CACHE
/* Cache a newly created datatype.*/
OSHMPI_STATIC_INLINE_PREFIX void strided_set_dtype_cache(size_t nelems, ptrdiff_t stride,
                                                         MPI_Datatype mpi_type,
                                                         size_t required_ext_nelems,
                                                         MPI_Datatype strided_type)
{
    OSHMPI_dtype_cache_obj_t *dobj = NULL;

    dobj = OSHMPIU_malloc(sizeof(OSHMPI_dtype_cache_obj_t));
    OSHMPI_ASSERT(dobj);

    /* Set new comm */
    dobj->nelems = nelems;
    dobj->stride = stride;
    dobj->dtype = mpi_type;
    dobj->ext_nelems = required_ext_nelems;
    dobj->sdtype = strided_type;

    OSHMPI_THREAD_ENTER_CS(&OSHMPI_strided_dtype_cache.cs);
    /* Insert in head, O(1) */
    LL_PREPEND(OSHMPI_strided_dtype_cache.head, dobj);
    OSHMPI_strided_dtype_cache.nobjs++;
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_strided_dtype_cache.cs);
}

/* Find if cached datatype already exists. */
OSHMPI_STATIC_INLINE_PREFIX int strided_find_dtype_cache(size_t nelems, ptrdiff_t stride,
                                                         MPI_Datatype mpi_type,
                                                         size_t required_ext_nelems,
                                                         MPI_Datatype * strided_type)
{
    int found = 0;
    OSHMPI_dtype_cache_obj_t *dobj = NULL;

    /* TODO: optimize search operation */
    OSHMPI_THREAD_ENTER_CS(&OSHMPI_strided_dtype_cache.cs);
    dobj = OSHMPI_strided_dtype_cache.head;
    LL_FOREACH(OSHMPI_strided_dtype_cache.head, dobj) {
        if (dobj->nelems == nelems && dobj->stride == stride
            && dobj->dtype == mpi_type && dobj->ext_nelems == required_ext_nelems) {
            found = 1;
            *strided_type = dobj->sdtype;
            break;
        }
    }
    OSHMPI_THREAD_EXIT_CS(&OSHMPI_strided_dtype_cache.cs);
    return found;
}
#endif

/* Create derived datatype for strided data format.
 * If it is contig (stride == 1), then the basic datatype is returned.
 * The caller must check the returned datatype to free it when necessary. */
OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_create_strided_dtype(size_t nelems, ptrdiff_t stride,
                                                             MPI_Datatype mpi_type,
                                                             size_t required_ext_nelems,
                                                             size_t * strided_cnt,
                                                             MPI_Datatype * strided_type)
{
    /* TODO: check non-int inputs exceeds int limit */

    /* Fast path: contig */
    if (stride == 1) {
        *strided_type = mpi_type;
        *strided_cnt = nelems;

        OSHMPI_DBGMSG("strided[%ld,%ld,0x%lx,%ld]=>sdtype[0x%lx,%ld].\n",
                      nelems, stride, (unsigned long) mpi_type, required_ext_nelems,
                      (unsigned long) *strided_type, *strided_cnt);
        return;
    }
#ifdef OSHMPI_ENABLE_STRIDED_DTYPE_CACHE
    /* Fast path: return a cached datatype if found */
    if (strided_find_dtype_cache(nelems, stride, mpi_type, required_ext_nelems, strided_type)) {
        OSHMPI_DBGMSG("strided[%ld,%ld,0x%lx,%ld]=>cached sdtype[0x%lx,1] returned.\n",
                      nelems, stride, (unsigned long) mpi_type, required_ext_nelems,
                      (unsigned long) *strided_type);
        *strided_cnt = 1;
        return;
    }
#endif

    /* Slow path: create a new datatype and cache it */
    MPI_Datatype vtype = MPI_DATATYPE_NULL;
    size_t elem_bytes = 0;

    OSHMPI_CALLMPI(MPI_Type_vector((int) nelems, 1, (int) stride, mpi_type, &vtype));

    /* Vector does not count stride after last chunk, thus we need to resize to
     * cover it when multiple elements with the stride_datatype may be used (i.e., alltoalls).
     * Extent can be negative in MPI, however, we do not expect such case in OSHMPI.
     * Thus skip any negative one */
    if (required_ext_nelems > 0) {
        if (mpi_type == OSHMPI_MPI_COLL32_T)
            elem_bytes = 4;
        else
            elem_bytes = 8;
        OSHMPI_CALLMPI(MPI_Type_create_resized
                       (vtype, 0, required_ext_nelems * elem_bytes, strided_type));
    } else
        *strided_type = vtype;
    OSHMPI_CALLMPI(MPI_Type_commit(strided_type));
    if (required_ext_nelems > 0)
        OSHMPI_CALLMPI(MPI_Type_free(&vtype));
    *strided_cnt = 1;

#ifdef OSHMPI_ENABLE_STRIDED_DTYPE_CACHE
    strided_set_dtype_cache(nelems, stride, mpi_type, required_ext_nelems, *strided_type);
#endif
    OSHMPI_DBGMSG("new strided[%ld,%ld,0x%lx,%ld]=>sdtype[0x%lx,1] created.\n",
                  nelems, stride, (unsigned long) mpi_type, required_ext_nelems,
                  (unsigned long) *strided_type);
}

OSHMPI_STATIC_INLINE_PREFIX void OSHMPI_free_strided_dtype(MPI_Datatype mpi_type,
                                                           MPI_Datatype * strided_type)
{
#ifdef OSHMPI_ENABLE_STRIDED_DTYPE_CACHE
    /* free at finalize */
#else
    if (mpi_type != *strided_type)
        OSHMPI_CALLMPI(MPI_Type_free(strided_type));
#endif
}
#endif /* INTERNAL_STRIDED_IMPL_H */

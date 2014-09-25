#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <omp.h> // timing only
#include <pmi.h>
#include <dmapp.h>

static inline void DMAPP_CHECK(dmapp_return_t rc, int line)
{
    if (DMAPP_RC_SUCCESS != rc) {
        const char* msg;
        dmapp_explain_error(rc, &msg);
        fprintf(stderr, "line %d: %s\n", line, msg);
        fflush(0);
        abort();
    }                                       
    return;
}

dmapp_jobinfo_t _job;
dmapp_seg_desc_t * _sheap = NULL;

int shmem_my_pe(void) { return (int)_job.pe; }
int shmem_n_pes(void) { return (int)_job.npes; }

void shmem_init(void)
{
    dmapp_return_t rc;

    /* Set the RMA parameters. */
    dmapp_rma_attrs_t rma_args={0};
    rma_args.put_relaxed_ordering = DMAPP_ROUTING_ADAPTIVE;
    rma_args.max_outstanding_nb   = DMAPP_DEF_OUTSTANDING_NB;
    rma_args.offload_threshold    = DMAPP_OFFLOAD_THRESHOLD;
    rma_args.max_concurrency = 1;

    printf("DMAPP_DEF_OUTSTANDING_NB = %d\n", DMAPP_DEF_OUTSTANDING_NB);

    /* Initialize DMAPP. */
    dmapp_rma_attrs_t actual_args={0};
    rc = dmapp_init(&rma_args, &actual_args);
    DMAPP_CHECK(rc,__LINE__);

    /* Get job related information. */
    rc = dmapp_get_jobinfo(&_job);
    DMAPP_CHECK(rc,__LINE__);

    _sheap = &(_job.sheap_seg);

    return;
}

void shmem_exit(int code)
{
    dmapp_return_t rc = dmapp_finalize();
    DMAPP_CHECK(rc,__LINE__);
    exit(code);
}

void * shmalloc(size_t bytes)
{
    void * ptr = dmapp_sheap_malloc(bytes);
    if (ptr==NULL) DMAPP_CHECK(DMAPP_RC_NO_SPACE,__LINE__);
    return ptr;
}

void shfree(void * ptr)
{
    dmapp_sheap_free(ptr);
    return;
}

void shmem_barrier_all(void)
{
    PMI_Barrier();
    return;
}

void shmem_double_get(double * target, const double * source, size_t nelems, int pe)
{
    dmapp_return_t rc = dmapp_get(target, (double*)source, _sheap, pe, nelems, DMAPP_QW);
    DMAPP_CHECK(rc,__LINE__);
    return;
}

void shmem_double_put(double * target, const double * source, size_t nelems, int pe)
{
    dmapp_return_t rc = dmapp_put(target, _sheap, pe, (double*)source, nelems, DMAPP_QW);
    DMAPP_CHECK(rc,__LINE__);
    return;
}

void shmem_double_iget(double *target, const double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
    dmapp_return_t rc = dmapp_iget(target, (double*)source, _sheap, pe, tst, sst, nelems, DMAPP_QW);
    DMAPP_CHECK(rc,__LINE__);
    return;
}

void shmem_double_iput(double *target, const double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
    dmapp_return_t rc = dmapp_iput(target, _sheap, pe, (double*)source, tst, sst, nelems, DMAPP_QW);
    DMAPP_CHECK(rc,__LINE__);
    return;
}

void shmemx_double_aget(double * dest, const double * src,
                        ptrdiff_t dstr, ptrdiff_t sstr,
                        size_t blksz, size_t blkct, int pe)
{
    double       *dtmp = dest;
    const double *stmp = src;
    if (blksz<blkct) {
        for (size_t i=0; i<blksz; i++) {
            shmem_double_iget(dtmp, stmp, dstr, sstr, blkct, pe);
            dtmp++; stmp++;
        }
    } else {
        for (size_t i=0; i<blkct; i++) {
            shmem_double_get(dtmp, stmp, blksz, pe);
            dtmp += dstr; stmp += sstr;
        }
    }
    return;
}

void shmemx_double_aput(double * dest, const double * src,
                        ptrdiff_t dstr, ptrdiff_t sstr,
                        size_t blksz, size_t blkct, dmapp_pe_t pe)
{
    //dmapp_syncid_handle_t syncid;
    double       *dtmp = dest;
    const double *stmp = src;
    if (blksz<=blkct) {
        for (size_t i=0; i<blksz; i++) {
            //dmapp_return_t rc = dmapp_iput_nb(dtmp, _sheap, pe, (double*)stmp, dstr, sstr, blkct, DMAPP_QW, &syncid);
            //dmapp_return_t rc = dmapp_iput_nbi(dtmp, _sheap, pe, (double*)stmp, dstr, sstr, blkct, DMAPP_QW);
            dmapp_return_t rc = dmapp_iput(dtmp, _sheap, pe, (double*)stmp, dstr, sstr, blkct, DMAPP_QW);
            DMAPP_CHECK(rc,__LINE__);
            dtmp++; stmp++;
        }
    } else {
        for (size_t i=0; i<blkct; i++) {
            //dmapp_return_t rc = dmapp_put_nb((void*)dtmp, _sheap, (dmapp_pe_t)pe, (void*)stmp, blksz, DMAPP_QW, &syncid);
            //dmapp_return_t rc = dmapp_put_nbi((void*)dtmp, _sheap, (dmapp_pe_t)pe, (void*)stmp, blksz, DMAPP_QW);
            dmapp_return_t rc = dmapp_put((void*)dtmp, _sheap, (dmapp_pe_t)pe, (void*)stmp, blksz, DMAPP_QW);
            DMAPP_CHECK(rc,__LINE__);
            dtmp += dstr; stmp += sstr;
        }
    }
    {
        //dmapp_return_t rc = dmapp_syncid_wait(&syncid);
        //dmapp_return_t rc = dmapp_gsync_wait();
        dmapp_return_t rc = DMAPP_RC_SUCCESS;
        DMAPP_CHECK(rc,__LINE__);
    }
    return;
}

void array_memzero(double * x, size_t n)
{
    for (size_t i=0; i<n; i++) {
        x[i] = 0.0;
    }
    return;
}

void array_meminit(double * x, size_t n)
{
    for (size_t i=0; i<n; i++) {
        x[i] = (double)i+1;
    }
    return;
}

int main(int argc, char* argv[])
{
    shmem_init();

    int mype = shmem_my_pe();
    int npes = shmem_n_pes();

    if (npes<2) {
        printf("This test requires more than 1 PE\n");
        fflush(0);
        exit(1);
    }

    int dim = (argc>1) ? atoi(argv[1]) : 8;

    /* circulant shift */
    int otherpe = (mype+1)%npes;

    double * distmat = shmalloc( dim*dim*sizeof(double) );
    shmem_barrier_all(); /* Cray SHMEM may not barrier above... */
    assert(distmat!=NULL);

    double * locmat = malloc( dim*dim*sizeof(double) );
    assert(locmat!=NULL);

    /* basic verification that things are working */

    array_meminit(locmat, dim*dim);
    shmem_double_put(distmat, locmat, dim*dim, otherpe);
    shmem_barrier_all();

    array_memzero(locmat, dim*dim);
    shmem_double_get(locmat, distmat, dim*dim, otherpe);
    shmem_barrier_all();

    if (mype==0) {
        for (int i=0; i<dim; i++) {
            printf("A[%d,*] = ", i);
            for (int j=0; j<dim; j++) {
                printf("%lf ", locmat[i*dim+j]);
            }
            printf("\n");
        }
    }
    fflush(stdout);
    shmem_barrier_all();

    /* submatrix verification */

    array_memzero(locmat, dim*dim);
    shmem_double_put(distmat, locmat, dim*dim, otherpe);
    shmem_barrier_all();

    double * submat = malloc( (dim/2)*(dim/2)*sizeof(double) );
    for (int i=0; i<dim/2; i++) {
        for (int j=0; j<dim/2; j++) {
            submat[i*dim/2+j] = i*dim/2+j+1;
        }
    }
    double t0 = omp_get_wtime();
    shmemx_double_aput(&(distmat[dim/4*dim+dim/4]), submat, dim, dim/2, dim/2, dim/2, otherpe);
    double t1 = omp_get_wtime();
    shmem_barrier_all();

    printf("time = %lf\n", t1-t0);

    array_memzero(locmat, dim*dim);
    shmem_double_get(locmat, distmat, dim*dim, otherpe);
    shmem_barrier_all();

    if (mype==0 && dim<12) {
        for (int i=0; i<dim; i++) {
            printf("B[%d,*] = ", i);
            for (int j=0; j<dim; j++) {
                printf("%lf ", locmat[i*dim+j]);
            }
            printf("\n");
        }
    }
    fflush(stdout);
    shmem_barrier_all();

    /*****************************/

    free(submat);
    free(locmat);
    shfree(distmat);

    shmem_exit(0);

    return 0;
}

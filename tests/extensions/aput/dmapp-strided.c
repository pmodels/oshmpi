#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <pmi.h>
#include <dmapp.h>

#define DMAPP_CHECK(rc)
/*
    do {                                        \
        if (DMAPP_RC_SUCCESS != rc) {           \
            char* msg;                          \
            dmapp_explain_error(rc, &msg);      \
            fprintf(stderr, "%s\n", msg);       \
            dmapp_finalize();                   \
            exit(rc);                           \
        } while(0);
*/

#define DIM 8

int _mype, _npes;
dmapp_seg_desc_t * _sheap = NULL;

int shmem_my_pe(void) { return _mype; }
int shmem_n_pes(void) { return _npes; }

void shmem_init(void)
{
    dmapp_return_t rc;

    /* Set the RMA parameters. */
    dmapp_rma_attrs_t rma_args={0};
    rma_args.put_relaxed_ordering = DMAPP_ROUTING_ADAPTIVE;
    rma_args.max_outstanding_nb   = DMAPP_DEF_OUTSTANDING_NB;
    rma_args.offload_threshold    = DMAPP_OFFLOAD_THRESHOLD;
    rma_args.max_concurrency = 1;

    /* Initialize DMAPP. */
    dmapp_rma_attrs_t actual_args={0};
    rc = dmapp_init(&rma_args, &actual_args);
    DMAPP_CHECK(rc);

    /* Get job related information. */
    dmapp_jobinfo_t job;
    rc = dmapp_get_jobinfo(&job);
    DMAPP_CHECK(rc);

    _mype = (int)job.pe;
    _npes = (int)job.npes;
    _sheap = &(job.sheap_seg);

    return;
}

void shmem_exit(int code)
{
    dmapp_return_t rc = dmapp_finalize();
    DMAPP_CHECK(rc);
    exit(code);
}

void * shmalloc(size_t bytes)
{
    void * ptr = dmapp_sheap_malloc(bytes);
    if (ptr==NULL) DMAPP_CHECK(DMAPP_RC_NO_SPACE);
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
    dmapp_return_t rc = dmapp_get((void*)target, (void*)source, _sheap, (dmapp_pe_t)pe, nelems, DMAPP_C_DOUBLE);
    DMAPP_CHECK(rc);
    return;
}

void shmem_double_put(double * target, const double * source, size_t nelems, int pe)
{
    dmapp_return_t rc = dmapp_put((void*)target, _sheap, (dmapp_pe_t)pe, (void*)source, nelems, DMAPP_C_DOUBLE);
    DMAPP_CHECK(rc);
    return;
}

void shmemx_double_aget(double * dest, const double * src,
                        ptrdiff_t dstr, ptrdiff_t sstr,
                        size_t blksz, size_t blkct, int pe)
{
    double       *dtmp = dest;
    const double *stmp = src;
    if (0 && blksz<blkct) {
        for (size_t i=0; i<blksz; i++) {
            //shmem_double_iget(dtmp, stmp, dstr, sstr, blkct, pe);
            dtmp++; stmp++;
        }
    } else {
        for (size_t i=0; i<blkct; i++) {
            //shmem_double_get(dtmp, stmp, blksz, pe);
            dtmp += dstr; stmp += sstr;
        }
    }
    return;
}

void shmemx_double_aput(double * dest, const double * src,
                        ptrdiff_t dstr, ptrdiff_t sstr,
                        size_t blksz, size_t blkct, dmapp_pe_t pe)
{
    dmapp_syncid_handle_t syncid;
    double       *dtmp = dest;
    const double *stmp = src;
    if (0 && blksz<=blkct) {
        for (size_t i=0; i<blksz; i++) {
            //shmem_double_iput(dtmp, stmp, dstr, sstr, blkct, pe);
            dtmp++; stmp++;
        }
    } else {
        for (size_t i=0; i<blkct; i++) {
            int rc = dmapp_put_nb((void*)dtmp, _sheap, (dmapp_pe_t)pe, (void*)stmp, blksz, DMAPP_C_DOUBLE, &syncid);
            DMAPP_CHECK(rc);
            dtmp += dstr; stmp += sstr;
        }
    }
    {
        int rc = dmapp_syncid_wait(&syncid);
        DMAPP_CHECK(rc);
    }
    return;
}

void array_memset(double * x, double val, int special)
{
    if (special==1) {
        for (int i=0; i<DIM; i++) {
            for (int j=0; j<DIM; j++) {
                x[i*DIM+j] = (double)i*DIM+j+1;
            }
        }
    } else {
        for (int i=0; i<DIM; i++) {
            for (int j=0; j<DIM; j++) {
                x[i*DIM+j] = val;
            }
        }
    }
    return;
}

int main(int argc, char* argv[])
{
    shmem_init();

    int mype = shmem_my_pe();
    int npes = shmem_n_pes();

    if (npes<2) exit(1);

    /* circulant shift */
    int otherpe = (mype+1)%npes;

    double * distmat = shmalloc( DIM*DIM*sizeof(double) );
    shmem_barrier_all(); /* Cray SHMEM may not barrier above... */
    assert(distmat!=NULL);

    double * locmat = malloc( DIM*DIM*sizeof(double) );
    assert(locmat!=NULL);

    /* basic verification that things are working */

    array_memset(locmat, 0.0, 1);
    shmem_double_put(distmat, locmat, DIM*DIM, otherpe);
    shmem_barrier_all();

    array_memset(locmat, 0.0, 0);
    shmem_double_get(locmat, distmat, DIM*DIM, otherpe);
    shmem_barrier_all();

    if (mype==0) {
        for (int i=0; i<DIM; i++) {
            printf("A[%d,*] = ", i);
            for (int j=0; j<DIM; j++) {
                printf("%lf ", locmat[i*DIM+j]);
            }
            printf("\n");
        }
    }
    fflush(stdout);
    shmem_barrier_all();

    /* submatrix verification */

    array_memset(locmat, 0.0, 0);
    shmem_double_put(distmat, locmat, DIM*DIM, otherpe);
    shmem_barrier_all();

    double * submat = malloc( (DIM/2)*(DIM/2)*sizeof(double) );
    for (int i=0; i<DIM/2; i++) {
        for (int j=0; j<DIM/2; j++) {
            submat[i*DIM/2+j] = i*DIM/2+j+1;
        }
    }
    //shmemx_double_aput(&(distmat[DIM/4*DIM+DIM/4]), submat, DIM, DIM/2, 4, 4, otherpe);
    shmem_barrier_all();

    array_memset(locmat, 0.0, 0);
    shmem_double_get(locmat, distmat, DIM*DIM, otherpe);
    shmem_barrier_all();

    if (mype==0) {
        for (int i=0; i<DIM; i++) {
            printf("B[%d,*] = ", i);
            for (int j=0; j<DIM; j++) {
                printf("%lf ", locmat[i*DIM+j]);
            }
            printf("\n");
        }
    }
    fflush(stdout);
    shmem_barrier_all();

    /*****************************/

    free(submat);
    //free(locmat);
    shfree(distmat);

    shmem_exit(0);

    return 0;
}

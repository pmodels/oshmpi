#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <dmapp.h>

#ifdef CHECK_RETURN_CODES
#define DMAPP_CHECK(a)                          \
    do {                                        \
        int rc = (a);                           \
        if (DMAPP_RC_SUCCESS != rc) {           \
            char* msg;                          \
            dmapp_explain_error(rc, &msg);      \
            fprintf(stderr, "%s\n", msg);       \
            dmapp_finalize();                   \
            exit(rc);                           \
        } while(0)
#else
#define DMAPP_CHECK(a) (a)
#endif

#define DIM 8

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
            DMAPP_CHECK( dmapp_put_nb(dtmp, sheap_segment, pe, stmp, blksz, DMAPP_C_DOUBLE, &syncid) );
            dtmp += dstr; stmp += sstr;
        }
    }
    DMAPP_CHECK( dmapp_syncid_wait(&syncid) );
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
    DMAPP_CHECK( dmapp_init_ext(NULL, &attr); );
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
    shmemx_double_aput(&(distmat[DIM/4*DIM+DIM/4]), submat, DIM, DIM/2, 4, 4, otherpe);
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

    DMAPP_CHECK( dmapp_finalize(); );

    return 0;
}

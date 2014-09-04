#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <shmem.h>

#define DIM 8

int evil;

void shmemx_double_aget(double * dest, const double * src,
                        ptrdiff_t dstr, ptrdiff_t sstr,
                        size_t blksz, size_t blkct, int pe)
{
    double *dtmp = dest, *stmp = src;
    if (dbsz<dstr || evil==1) {
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
                        size_t blksz, size_t blkct, int pe)
{
    double *dtmp = dest, *stmp = src;
    if (dbsz<dstr || evil==1) {
        for (size_t i=0; i<blksz; i++) {
            shmem_double_iput(dtmp, stmp, dstr, sstr, blkct, pe);
            dtmp++; stmp++;
        }
    } else {
        for (size_t i=0; i<blkct; i++) {
            shmem_double_put(dtmp, stmp, blksz, pe);
            dtmp += dstr; stmp += sstr;
        }
    }
    return;
}

void array_memset(double * x, double val, int special)
{
    if (special==1) {
        for (int i=0; i<DIM; i++) {
            for (int j=0; j<DIM; j++) {
                x[i*DIM+j] = (double)i*DIM+j;
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
    start_pes(0);
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
    shmem_double_get(locmat, destmat, DIM*DIM, otherpe);
    shmem_barrier_all();

    for (int i=0; i<DIM; i++) {
        for (int j=0; j<DIM; j++) {
            printf("%d: x[%d,%d] = %lf\n", mype, i, j, x[i*DIM+j]);
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
            submat[i*DIM+j] = i*DIM+j;
        }
    }
    shmemx_double_aput(distmat, locmat, DIM, DIM/2, 4, 4, otherpe);
    shmem_barrier_all();

    array_memset(locmat, 0.0, 0);
    shmem_double_get(locmat, destmat, DIM*DIM, otherpe);
    shmem_barrier_all();

    for (int i=0; i<DIM; i++) {
        for (int j=0; j<DIM; j++) {
            printf("%d: x[%d,%d] = %lf\n", mype, i, j, x[i*DIM+j]);
        }
    }
    fflush(stdout);
    shmem_barrier_all();

    /*****************************/

    free(submat);
    free(locmat);
    shfree(distmat);

    return 0;
}

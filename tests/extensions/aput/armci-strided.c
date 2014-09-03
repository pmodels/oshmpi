#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <shmem.h>

/* upcast to ULL to prevent overflow in some weird case, e.g. DIM*DIM*DIM*N */
#define DIM 10ULL

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
                x[i*DIM+j] = (double)i*DIM+j + mype/100.0;
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

    double * distmat = shmalloc( DIM*DIM*sizeof(double) );
    assert(distmat!=NULL);

    double * locmat = malloc( DIM*DIM*sizeof(double) );
    assert(locmat!=NULL);

    array_memset(locmat, 0.0, 1);
    shmem_double_put(distmat, locmat, DIM*DIM, mype);
    shmem_barrier_all();

    array_memset(locmat, 0.0, 0);
    shmem_double_get(locmat, destmat, DIM*DIM, mype);
    shmem_barrier_all();

    for (int i=0; i<DIM; i++) {
        for (int j=0; j<DIM; j++) {
            printf("%d: x[%d,%d] = %lf\n", mype, i, j, x[i*DIM+j]);
        }
    }

    free(locmat);
    shfree(distmat);

    return 0;
}

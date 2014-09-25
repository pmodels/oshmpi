#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <omp.h> // timing only
#include <shmem.h>

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
                        size_t blksz, size_t blkct, int pe)
{
    double       *dtmp = dest;
    const double *stmp = src;
    if (blksz<=blkct) {
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
    //start_pes(0);
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

    if (mype==0 && dim<15) {
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

    /* submatrix verification for aput */
    {
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

        printf("%d: aput time = %lf\n", mype, t1-t0);

        array_memzero(locmat, dim*dim);
        shmem_double_get(locmat, distmat, dim*dim, otherpe);
        shmem_barrier_all();

        if (mype==0 && dim<15) {
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

        free(submat);
    }

    /* submatrix verification for aget */
    {
        array_meminit(locmat, dim*dim);
        shmem_double_put(distmat, locmat, dim*dim, otherpe);
        shmem_barrier_all();

        double * submat = malloc( (dim/2)*(dim/2)*sizeof(double) );

        double t0 = omp_get_wtime();
        shmemx_double_aget(submat, &(distmat[dim/4*dim+dim/4]), dim/2, dim, dim/2, dim/2, otherpe);
        double t1 = omp_get_wtime();
        shmem_barrier_all();

        printf("%d: aget time = %lf\n", mype, t1-t0);

        if (mype==0 && dim<15) {
            for (int i=0; i<dim/2; i++) {
                printf("B[%d,*] = ", i);
                for (int j=0; j<dim/2; j++) {
                    printf("%lf ", submat[i*dim/2+j]);
                }
                printf("\n");
            }
        }
        fflush(stdout);
        shmem_barrier_all();

        free(submat);
    }

    /*****************************/

    free(locmat);
    shfree(distmat);

    shmem_finalize();

    return 0;
}

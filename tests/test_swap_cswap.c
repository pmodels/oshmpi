#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <shmem.h>

int main(void)
{
    start_pes(0);

    int mype = shmem_my_pe();
    int npes = shmem_n_pes();

    int target = (mype+1) % npes;

    int       * shi  = shmalloc(sizeof(int));
    //long      * shl  = shmalloc(sizeof(long));
    //long long * shll = shmalloc(sizeof(long long));
    float     * shf  = shmalloc(sizeof(float));
    double    * shd  = shmalloc(sizeof(double));

    shi[0]  = 1;
    //shl[0]  = 1L;
    //shll[0] = 1LL;
    shf[0]  = 1.0f;
    shd[0]  = 1.0;

    int       ir  = 0;
    //long      lr  = 0L;
    //long long llr = 0LL;
    float     fr  = 0.0f;
    double    dr  = 0.0;

    shmem_barrier_all();

    ir  = shmem_int_swap(     shi,  2,     target);
    //lr  = shmem_long_swap(    shl,  2L,    target);
    //llr = shmem_longlong_swap(shll, 2LL,   target);
    fr  = shmem_float_swap(   shf,  2.0f,  target);
    dr  = shmem_double_swap(  shd,  2.0,   target);

    printf("ir  = %d\n",   ir);
    //printf("lr  = %ld\n",  lr);
    //printf("llr = %lld\n", llr);
    printf("fr  = %f\n",   fr);
    printf("dr  = %lf\n",  dr);
    fflush(stdout);

    assert(ir==1);
    //assert(lr==1L);
    //assert(llr==1LL);
    assert(fabs(fr-1.0f)<1.0e-6);
    assert(fabs(dr-1.0)<1.0e-12);

    shmem_barrier_all();

    ir  = shmem_int_swap(     shi,  0,       target);
    //lr  = shmem_long_swap(    shl,  0L,      target);
    //llr = shmem_longlong_swap(shll, 0LL,     target);
    fr  = shmem_float_swap(   shf,  0.0f,    target);
    dr  = shmem_double_swap(  shd,  0.0,     target);

    printf("ir  = %d\n",   ir);
    //printf("lr  = %ld\n",  lr);
    //printf("llr = %lld\n", llr);
    printf("fr  = %f\n",   fr);
    printf("dr  = %lf\n",  dr);
    fflush(stdout);

    assert(ir==1);
    //assert(lr==1L);
    //assert(llr==1LL);
    assert(fabs(fr-1.0f)<1.0e-6);
    assert(fabs(dr-1.0)<1.0e-12);

    shmem_barrier_all();

    shfree(shd);
    shfree(shf);
    //shfree(shll);
    //shfree(shl);
    shfree(shi);

    shmem_barrier_all();

    if (mype==0) {
        printf("SUCCESS \n");
    }
    return 0;
}

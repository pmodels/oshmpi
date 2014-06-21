#include <stdio.h>
#include <shmem.h>

int main(void)
{
    start_pes(0);

    int mype = shmem_my_pe();
    int npes = shmem_n_pes();

    int       * shi  = shmalloc(sizeof(int));
    long      * shl  = shmalloc(sizeof(long));
    long long * shll = shmalloc(sizeof(long long));

    shi[0]  = 0;
    shl[0]  = 0L;
    shll[0] = 0LL;

    int       ir  = 0;
    long      lr  = 0L;
    long long llr = 0LL;

    shmem_barrier_all();

    shmem_int_inc(shi, 0);
    shmem_long_inc(shl, 0);
    shmem_longlong_inc(shll, 0);

    //shmem_fence();
    shmem_barrier_all();

    ir  = shmem_int_finc(shi, 0);
    lr  = shmem_long_finc(shl, 0);
    llr = shmem_longlong_finc(shll, 0);

    shmem_barrier_all();

    printf("ir = %d\n");
    printf("lr = %ld\n");
    printf("llr = %lld\n");

    shmem_barrier_all();

    assert(ir==(int)npes);
    assert(lr==(long)npes);
    assert(llr==(long long)npes);

    shmem_fence();

    shmem_int_add(shi, 1000, 0);
    shmem_long_add(shl, 1000L, 0);
    shmem_longlong_add(shll, 1000LL, 0);

    shmem_fence();

    shmem_int_add(shi, 1000, 0);
    shmem_long_add(shl, 1000L, 0);
    shmem_longlong_add(shll, 1000LL, 0);

    shmem_barrier_all();

    shfree(shll);
    shfree(shl);
    shfree(shi);

    shmem_barrier_all();

    printf("SUCCESS \n");
    return 0;
}

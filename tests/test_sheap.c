#include <stdio.h>

#include <shmem.h>

#define SIZE 100

int main(void)
{
    start_pes(0);
    int mype = shmem_my_pe();
    int npes = shmem_n_pes();

    int * in  = shmalloc(SIZE*sizeof(int));
    int * out = shmalloc(SIZE*sizeof(int));

    int source = 0;
    int target = (mype+1)%npes;

    for (int i=0; i<SIZE; i++)
        in [i] = 1+mype;

    for (int i=0; i<SIZE; i++)
        out[i] = -(1+mype);

    shmem_barrier_all();

    if (mype==source) {
        printf("in  = %p \n", in );
        printf("out = %p \n", out);

        printf("before shmem_int_put \n");
        shmem_int_put(out, in , (size_t)SIZE, target);
    }

    printf("before shmem_barrier_all \n");
    shmem_barrier_all();

    if (mype==target) {
        for (int i=0; i<SIZE; i++)
            if (out[i] != (1+source))
                printf("%d: element %d: correct = %d, got %d \n", mype, i, (1+source), out[i]);
    }

    printf("before shmem_barrier_all \n");
    shmem_barrier_all();

    for (int i=0; i<SIZE; i++)
        in [i] = 1+mype;

    for (int i=0; i<SIZE; i++)
        out[i] = -(1+mype);

    if (mype==source) {
        printf("in  = %p \n", in );
        printf("out = %p \n", out);

        printf("before shmem_int_get \n");
        shmem_int_get(out, in , (size_t)SIZE, target);
    }

    printf("before shmem_barrier_all \n");
    shmem_barrier_all();

    if (mype==target) {
        for (int i=0; i<SIZE; i++)
            if (out[i] != (1+source))
                printf("%d: element %d: correct = %d, got %d \n", mype, i, (1+source), out[i]);
    }

    printf("before shmem_barrier_all \n");
    shmem_barrier_all();

    printf("test finished \n");
    return 0;
}

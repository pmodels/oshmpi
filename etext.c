#include <stdio.h>

#include <shmem.h>

#define SIZE 10000
static int source[SIZE];
static int target[SIZE];

int main(void)
{
    start_pes(0);
    int mype = shmem_my_pe();
    int npes = shmem_n_pes();

    for (int i=0; i<SIZE; i++)
        source[i] = 1+mype;

    for (int i=0; i<SIZE; i++)
        target[i] = -(1+mype);

    shmem_barrier_all();

    printf("source = %p \n", source);
    printf("target = %p \n", target);
    printf("source[0] = %d \n", source[0]);
    printf("target[0] = %d \n", target[0]);

    int them = (mype+1)%npes;
    printf("before shmem_int_put \n");
    shmem_int_put(target, source, (size_t)SIZE, them);
    printf("before shmem_barrier_all \n");
    shmem_barrier_all();
    them = (mype>0 ? mype-1 : npes-1);
    for (int i=0; i<SIZE; i++)
        if (target[i] != them)
            printf("PE %d, element %d: correct = %d, got %d \n", mype, i, them, target[i]);

    printf("%d: test finished \n", mype);
    return 0;
}

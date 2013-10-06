#include <stdio.h>

#include <shmem.h>

const int n = 1000000;
int source[n];
int target[n];

int main(void)
{
    start_pes(0);
    int mype = shmem_my_pe();
    int npes = shmem_n_pes();

    for (int i=0; i<n; i++)
        source[i] = mype;

    shmem_barrier_all();

    int them = (mype+1)%npes;
    shmem_int_put(target, source, (size_t)n, them);
    shmem_barrier_all();
    them = (mype>0 ? mype-1 : npes-1);
    for (int i=0; i<n; i++)
        if (target[i] != them)
            printf("PE %d, element %d: correct = %d, got %d \n", mype, i, them, target[i]);

    return 0;
}

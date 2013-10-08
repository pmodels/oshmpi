#include <stdio.h>

#include <shmem.h>

int main(void)
{
    start_pes(0);
    printf("I am %d of %d. \n", 
            shmem_my_pe(), shmem_n_pes() );

    return 0;
}

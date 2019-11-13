/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <shmem.h>

int main(int argc, char *argv[])
{
    shmem_init();

    int mype = shmem_my_pe();
    static int flag = 0;

    if (mype == 0) {
        shmem_int_atomic_set(&flag, 1, 1);
    } else if (mype == 1) {
        shmem_int_wait_until(&flag, SHMEM_CMP_EQ, 1);
    }

    if (mype == 1) {
        fprintf(stdout, "Passed\n");
        fflush(stderr);
    }

    shmem_finalize();
    return 0;
}

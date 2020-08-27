/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>

#if defined(SHMEM_REQUIRE_THREAD_SERIALIZED)
int required = SHMEM_THREAD_SERIALIZED;
const char *required_str = "SHMEM_THREAD_SERIALIZED";
#elif defined(SHMEM_REQUIRE_THREAD_FUNNELED)
int required = SHMEM_THREAD_FUNNELED;
const char *required_str = "SHMEM_THREAD_FUNNELED";
#elif defined(SHMEM_REQUIRE_THREAD_SINGLE)
int required = SHMEM_THREAD_SINGLE;
const char *required_str = "SHMEM_THREAD_SINGLE";
#else /* Default THREAD_MULTIPLE */
int required = SHMEM_THREAD_MULTIPLE;
const char *required_str = "SHMEM_THREAD_MULTIPLE";
#endif

int main(int argc, char *argv[])
{
    int provided = 0;
    int mype = -1;

    shmem_init_thread(required, &provided);

    if (provided < required) {
        fprintf(stderr, "Requested %s (0x%x), but provided 0x%x\n",
                required_str, required, provided);
        fflush(stderr);
        shmem_global_exit(-1);
    } else if (provided > required) {
        fprintf(stdout, "Requested %s (0x%x), lower than provided 0x%x\n",
                required_str, required, provided);
        fflush(stdout);
    }


    mype = shmem_my_pe();
    shmem_finalize();

    /* Check local routines only */
    if (mype == 0) {
        fprintf(stdout, "Passed\n");
        fflush(stderr);
    }

    return 0;
}

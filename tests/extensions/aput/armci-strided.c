#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <shmem.h>

int evil;

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
}

int main(int argc, char* argv[])
{





    return 0;
}

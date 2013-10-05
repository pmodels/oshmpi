#include <stdio.h>
#include <stdlib.h>

#  define unlikely(x_) __builtin_expect(!!(x_),0)
#  define likely(x_)   __builtin_expect(!!(x_),1)

int main(int argc, char * argv[])
{
    int i = 0;
    if(likely(argc>1)) {
        i = atoi(argv[1]);
    }

    printf("i = %d \n", i);

    return 0;
}

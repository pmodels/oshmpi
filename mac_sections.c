#include <stdio.h>
#include <stdlib.h>
#include <mach-o/getsect.h>

/* taken from http://stackoverflow.com/questions/1765969/unable-to-locate-definition-of-etext-edata-end */

int main(int argc, char *argv[])
{
    printf("    program text (etext)      %10p\n", (void*)get_etext());
    printf("    initialized data (edata)  %10p\n", (void*)get_edata());
    printf("    uninitialized data (end)  %10p\n", (void*)get_end());
    return 0;
}

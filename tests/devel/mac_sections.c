/* based on http://stackoverflow.com/questions/1765969/unable-to-locate-definition-of-etext-edata-end */
/* see http://stackoverflow.com/questions/10301542/getting-process-base-address-in-mac-osx to fix the remaining issues */
#if defined(__APPLE__)
#include <stdio.h>
#include <stdlib.h>
#include <mach-o/getsect.h>
int main(int argc, char *argv[])
{
    int e;
    int f=3333;
    static int g;
    static int h=4444;

    printf("program text (etext)      %p\n", (void*)get_etext());
    printf("initialized data (edata)  %p\n", (void*)get_edata());
    printf("uninitialized data (end)  %p\n", (void*)get_end());

    printf("&e=%p\n", &e);
    printf("&f=%p\n", &f);
    printf("&g=%p\n", &g);
    printf("&h=%p\n", &h);
    printf("This test is only for Apple Mac\n");
    return 0;
}
#else
int main(void) { return 0; }
#endif

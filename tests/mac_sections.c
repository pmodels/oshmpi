/* based on http://stackoverflow.com/questions/1765969/unable-to-locate-definition-of-etext-edata-end */

/* see http://stackoverflow.com/questions/10301542/getting-process-base-address-in-mac-osx to fix the remaining issues */

#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__ 
#include <mach-o/getsect.h>
#endif
int main(int argc, char *argv[])
{
#if defined(__APPLE__) 
    int e;
    int f=3333;
    static int g;
    static int h=4444;

    printf("program text (etext)      %p\n", (void*)get_etext());
    printf("initialized data (edata)  %p\n", (void*)get_edata());
    printf("uninitialized data (end)  %p\n", (void*)get_end());

    printf("&a=%p\n", &a);
    printf("&b=%p\n", &b);
    printf("&c=%p\n", &c);
    printf("&d=%p\n", &d);
    printf("&e=%p\n", &e);
    printf("&f=%p\n", &f);
    printf("&g=%p\n", &g);
    printf("&h=%p\n", &h);
#else
    printf("This test is only for Apple Mac\n");
#endif
    return 0;
}

#include <stdlib.h>
main(int argc, char *argv[])
{
    int b,* a ,c[4];
    do {
        a = malloc(sizeof(int)*4);
        a[0]=b*2;
        a[1]=b*3;
        a[2]=b*4;
        a[3]=b*5;
    } while(0);
    do {
        c[0]=b*2;
        c[1]=b*3;
        c[2]=b*4;
        c[3]=b*5;
    } while(0);
    return 0;
}

#include <stdlib.h>
#include <stdio.h>
int main(int argc, char *argv[])
{
    struct _ { int *a ; int c[4]; } __;
    int b = atoi(argv[1]);
    __.a = malloc(sizeof(int)*4);
    do {
        __.a[0]=b*2;
        __.a[1]=b*3;
        __.a[2]=b*4;
        __.a[3]=b*5;
    } while(0);
    do {
        __.c[0]=__.a[0]*2;
        __.c[1]=__.a[1]*3;
        __.c[2]=__.a[2]*4;
        __.c[3]=__.a[3]*5;
    } while(0);
    free(__.a);
    return __.c[0];
}

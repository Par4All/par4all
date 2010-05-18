#include <stdio.h>
static void pmax(int a, int b)
{
    if( a > b )
        return;
    printf("%d\n",a > b ? a : b);
}

int main(int argc, char **argv)
{
    int d=0;
    pmax(2,argc);
    return d;
}


#include <stdio.h>
static int pmax(int a, int b)
{
    int c =a > b ? a : b;
    printf("%d\n",c);
    return c;
}

int main(int argc, char **argv)
{
    pmax(2+3,argc);
    return 0;
}


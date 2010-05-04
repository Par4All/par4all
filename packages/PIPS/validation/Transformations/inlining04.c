#include <stdio.h>
static int pmax(int a, int b)
{
    int c =a > b ? a : b;
    printf("%d\n",c);
    return c;
}

int main(int argc, char **argv)
{
    return pmax(1,2) + pmax(2,3);
}


#include <stdlib.h>

#define N 20000000
float dotprod(float b[N], float c[N])
{
    int i;
    float a=0;
    for(i=0; i<N; ++i)
    {
        a += b[i] * c[i] ;
    }
    return a;
}

int main()
{
    float a, *b, *c;
    int i;
    b = malloc(N * sizeof(float));
    c = malloc(N * sizeof(float));
    for(i=0;i<N;i++)
    {
        b[i]=i;
        c[i]=N-i;
    }
    a=dotprod(b, c);
    return 0;
}


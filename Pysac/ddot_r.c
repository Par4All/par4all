#include <stdlib.h>
#include <stdio.h>

float ddot_r(int n,float b[n], float c[n])
{
    int i;
    float a=0;
    for(i=0; i<n; ++i)
        a += b[i] * c[i] ;
    return a;
}

int main(int argc, char ** argv)
{
    int i,n=argc==1?200:atoi(argv[1]);
    float a, (*b)[n], (*c)[n];
    b = malloc(n * sizeof(float));
    c = malloc(n * sizeof(float));
    for(i=0;i<n;i++)
    {
        (*b)[i]=(float)i;
        (*c)[i]=(float)i;
    }
    a=ddot_r(n,*b, *c);
    printf("%f",a);
    free(b);
    free(c);
    return 0;
}


#include <stdlib.h>
#include <stdio.h>

void ddot_ui(unsigned int n,unsigned int b[n], unsigned int c[n], unsigned int *r)
{
    unsigned int i;
    unsigned int a=0;
    for(i=0; i<n; ++i)
        a += b[i] * c[i] ;
    *r = a;
}

unsigned int main(unsigned int argc, char ** argv)
{
    unsigned int i,n=argc==1?200:atoi(argv[1]);
    unsigned int a, (*b)[n], (*c)[n];
    b = malloc(n * sizeof(unsigned int));
    c = malloc(n * sizeof(unsigned int));
    for(i=0;i<n;i++)
    {
        (*b)[i]=(unsigned int)i;
        (*c)[i]=(unsigned int)i;
    }
    ddot_ui(n,*b, *c, &a);
    printf("%d",a);
    free(b);
    free(c);
    return 0;
}


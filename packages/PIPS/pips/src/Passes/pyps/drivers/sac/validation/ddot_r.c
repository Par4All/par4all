#include <stdlib.h>
#include <stdio.h>
#include "tools.h"

float ddot_r(int n,float b[n], float c[n], float* a)
{
    int i;
    float r=0;
    for(i=0; i<n; ++i)
        r += b[i] * c[i] ;
    *a = r;
}

int main(int argc, char ** argv)
{
    int i,n;
    float a, *b, *c;
    init_args(argc, argv);
    n = atoi(argv[1]);
    b = (float*) malloc(n * sizeof(float));
    c = (float*) malloc(n * sizeof(float));
    init_data_float(b,n);
    init_data_float(c,n);
    close_data_file();
    ddot_r(n,b, c, &a);
    printf("%f\n",a);
    free(b);
    free(c);
    return 0;
}


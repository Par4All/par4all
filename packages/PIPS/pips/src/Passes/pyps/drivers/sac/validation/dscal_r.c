#include <stdio.h>
#include <stdlib.h>
#include "tools.h"

void dscal_r(int n,float da,float dx[n])
{
    int i;
    for (i = 0; i < n; i++)
        dx[i] = da*dx[i];
}

int main(int argc, char * argv[])
{
    int n;
    float *dx;
    int i;

    if (argc < 3)
    {
	    fprintf(stderr, "Usage: %s size data_file\n", argv[0]);
	    return 1;
    }
    n = atoi(argv[1]);
    dx = (float*) malloc(n*sizeof(float));
    init_data_file(argv[2]);
    init_data_float(dx, n);
    close_data_file();
    dscal_r(n,42.,dx);
    print_array_float("res", dx, n);
    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include "tools.h"

void dscal_ur(int n,float da,float dx[n])
{
    int i,m;
    m = n % 5;
    if (m != 0)
    {
        for (i = 0; i < m; i++)
            dx[i] = da*dx[i];
        if (n < 5)
            return;
    }
    for (i = m; i < n; i = i + 5)
    {
        dx[i] = da*dx[i];
        dx[i+1] = da*dx[i+1];
        dx[i+2] = da*dx[i+2];
        dx[i+3] = da*dx[i+3];
        dx[i+4] = da*dx[i+4];
    }
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
    dscal_ur(n,42.,dx);
    print_array_float("res", dx, n);
    return 0;
}


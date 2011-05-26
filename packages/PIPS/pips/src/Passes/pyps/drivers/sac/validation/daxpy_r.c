#include <stdio.h>
#include <stdlib.h>
#include "tools.h"


void daxpy_r(int n,float da,float dx[n],float dy[n])
{
    int i;
    /* code for both increments equal to 1 */

    for (i = 0;i < n; i++)
        dy[i] = dy[i] + da*dx[i];
}

int main(int argc, char * argv[])
{
    int n;
    float (*dx)[n],(*dy)[n];
    int i;
    if (argc < 3)
    {
	    fprintf(stderr, "Usage: %s size data_file\n", argv[0]);
	    return 1;
    }
    n = atoi(argv[1]);
    dx=malloc(sizeof(float)*n);
    dy=malloc(sizeof(float)*n);
    init_data_file(argv[2]);
    init_data_float(dx,n);
    init_data_float(dy,n);
    close_data_file();
    daxpy_r(n,42.,*dx,*dy);
    print_array_float("dy",dy,n);
    free(dx);
    free(dy);
    return 0;
}


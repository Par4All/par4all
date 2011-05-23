#include <stdio.h>
#include <stdlib.h>


void daxpy_ur(int n,float da,float dx[n],float dy[n])
{
    int i,m;
    m = n % 4;
    if ( m != 0)
    {
        for (i = 0; i < m; i++)
            dy[i] = dy[i] + da*dx[i];
        if (n < 4)
            return;
    }
    for (i = m; i < n; i = i + 4)
    {
        dy[i] = dy[i] + da*dx[i];
        dy[i+1] = dy[i+1] + da*dx[i+1];
        dy[i+2] = dy[i+2] + da*dx[i+2];
        dy[i+3] = dy[i+3] + da*dx[i+3];
    }
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
//    dx=malloc(sizeof(float)*n);
//    dy=malloc(sizeof(float)*n);
    posix_memalign(&dx, 32, sizeof(float)*n);
    posix_memalign(&dy, 32, sizeof(float)*n);
    init_data_file(argv[2]);
    init_data_float(dx,n);
    init_data_float(dy,n);
    close_data_file();
    daxpy_ur(n,.5*n,*dx,*dy);
    print_array_float("dy",dy,n);
    free(dx);
    free(dy);
    return 0;
}


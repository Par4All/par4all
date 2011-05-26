#include <stdio.h>
#include <stdlib.h>

#include "tools.h"

void matrix_mul_const(size_t N, float C[N][N], float A[N][N], float val) {
    size_t i,j;
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            C[i][j]=A[i][j] * val;
        }
    }
}

int main(int argc, char ** argv) {
    int i,j,n;
    float *a,*b;

    if (argc < 3)
    {
	    fprintf(stderr, "Usage: %s size data_file\n", argv[0]);
	    return 1;
    }
    n = atoi(argv[1]);
    a=malloc(sizeof(float)*n*n);
    b=malloc(sizeof(float)*n*n);
    init_data_file(argv[2]);
    init_data_float(a, n*n);
    close_data_file();
    matrix_mul_const(n,b,a,(float)n);
    print_array_float("b",b,n*n);
    free(a);
    free(b);
    return 0;
}

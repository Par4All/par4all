#include <stdio.h>
#include <stdlib.h>

#include "tools.h"

void matrix_mul_matrix(size_t N, float C[N][N], float A[N][N], float B[N][N]) {
    size_t i,j,k;
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            for(k=0;k<N;k++)
                C[i][j]+=A[i][k] * B[k][j];
        }
    }
}
int main(int argc, char ** argv) {

    int i,j,n;
    float *a,*b,*c;

    if (argc < 3)
    {
	    fprintf(stderr, "Usage: %s size data_file\n", argv[0]);
	    return 1;
    }
    n = atoi(argv[1]);
    a=malloc(sizeof(float)*4*n*4*n);
    b=malloc(sizeof(float)*n*4*n*4);
    c=calloc(n*4*n*4,sizeof(float));
    init_data_file(argv[2]);
    init_data_float(a, 4*4*n*n);
    init_data_float(b, 4*4*n*n);
    close_data_file();

    matrix_mul_matrix(n,c,a,b);

    print_array_float("c",c,n*n);

    return 0;
}

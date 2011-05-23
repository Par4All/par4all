#include <stdio.h>
#include <stdlib.h>

#include "tools.h"


void matrix_mul_vect(size_t N, float C[N], float A[N][N], float B[N]) {
    size_t i,j;
    for (i=0; i<N; i++) {
        C[i]=0;
        for (j=0; j<N; j++) {
            C[i]+=A[i][j] * B[j];
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
    a=malloc(sizeof(float)*n*n);
    b=malloc(sizeof(float)*n);
    c=malloc(sizeof(float)*n);

    init_data_file(argv[2]);
    init_data_float(a,n*n);
    init_data_float(b,n);
    close_data_file();
    matrix_mul_vect(n,c,a,b);

    print_array_float("c",c,n);
    free(a);
    free(b);
    free(c);

    return 0;
}

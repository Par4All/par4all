#include <stdio.h>
#include <stdlib.h>

void matrix_mul_matrix(size_t N, float C[4*N][4*N], float A[4*N][4*N], float B[4*N][4*N]) {
    size_t i,j,k;
    for (i=0; i<4*N; i++) {
        for (j=0; j<4*N; j++) {
            for(k=0;k<4*N;k++)
            {
                C[i][j]+=A[i][k] * B[k][j];
            }
        }
    }
}
int main(int argc, char ** argv) {
    int i,j,n = argc==1 ? 10 : atoi(argv[1]);
    float (*a)[n][n],(*b)[n][n],(*c)[n][n];
    a=malloc(sizeof(float)*16*n*n);
    b=malloc(sizeof(float)*16*n*n);
    c=calloc(sizeof(float),16*n*n);
    for(i=0;i<4*n;i++)
        for(j=0;j<4*n;j++)
            (*a)[i][j]=(*b)[i][j]=i*j;
    matrix_mul_matrix(n,*c,*a,*b);

    for(i=0;i<n;i++)
        printf("%f",(*b)[i][i]);
    free(a);
    free(b);
    return 0;
}

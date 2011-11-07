#include <stdio.h>
#include <stdlib.h>

void matrix_mul_matrix(size_t N, float C[N][N], float A[N][N], float B[N][N]) {
    size_t i,j,k;
a:  for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            C[i][j]=0;
            for(k=0;k<N;k++)
            {
                C[i][j]+=A[i][k] * B[k][j];
            }
        }
    }
}
int main(int argc, char ** argv) {
    int i,j,n = argc==1 ? 10 : atoi(argv[1]);
    float (*a)[n][n],(*b)[n][n],(*c)[n][n];
    a=malloc(sizeof(float)*n*n);
    b=malloc(sizeof(float)*n*n);
    c=malloc(sizeof(float)*n*n);
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            (*a)[i][j]=(*b)[i][j]=i*j;
    matrix_mul_matrix(n,*c,*a,*b);

    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            printf("%f",(*b)[i][j]);
    free(a);
    free(b);
    return 0;
}

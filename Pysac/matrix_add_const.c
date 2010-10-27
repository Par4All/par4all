#include <stdio.h>
#include <stdlib.h>
void matrix_add_const(size_t N, float C[N][N], float A[N][N], float val) {
    size_t i,j;
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            C[i][j]=A[i][j] + val;
        }
    }
}

int main(int argc, char ** argv) {
    int i,j,n = argc==1 ? 10 : atoi(argv[1]);
    float (*a)[n][n],(*b)[n][n];
    a=malloc(sizeof(float)*n*n);
    b=malloc(sizeof(float)*n*n);
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            (*a)[i][j]=i*j;
    matrix_add_const(n,*b,*a,(float)n);
    for(i=0;i<n;i++)
        printf("%f",(*b)[i][i]);
    free(a);
    free(b);
    return 0;
}

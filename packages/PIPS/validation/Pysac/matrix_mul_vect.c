#include <stdio.h>
#include <stdlib.h>


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
    int i,j,n = argc==1 ? 10 : atoi(argv[1]);
    float (*a)[n][n],(*b)[n], (*c)[n];
    a=malloc(sizeof(float)*n*n);
    b=malloc(sizeof(float)*n);
    c=malloc(sizeof(float)*n);
    for(i=0;i<n;i++) {
        (*b)[i]=1+i;
        for(j=0;j<n;j++)
            (*a)[i][j]=1+i*j;
    }
    matrix_mul_vect(n,c,a,b);

    for(i=0;i<n;i++)
        printf("|%f|",(*c)[i]);
    free(a);
    free(b);
    return 0;
}

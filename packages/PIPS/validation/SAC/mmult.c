#include <stdio.h>
#ifndef N
#define N 4
#endif

void Matrix_Mult(float a[N][N], float b[N][N], float c[N][N])
{
   int i, j, k;
loop0:
   for(i = 0; i <N; i ++)
loop1:
      for(j = 0; j <N; j ++) {
         c[i][j] = 0;
loop2:
         for(k = 0; k <N; k ++)
            c[i][j] = c[i][j]+a[i][k]*b[k][j];
      }
}
int main()
{
    float a[N][N], b[N][N], c[N][N];
    int i,j;
    for(i=0;i<N;i++)
        for(j=0;j<N;j++)
            a[i][j]=b[i][j]=i;

    Matrix_Mult(a,b,c);

    for(i=0;i<N;i++)
        for(j=0;j<N;j++)
            printf("%f-",c[i][j]);
    return 0;
}

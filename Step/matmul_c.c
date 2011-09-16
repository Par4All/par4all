#include <stdio.h>

int main(void)
{
  int n=10;
  float a[n][n], b[n][n], c[n][n];
  int i,j,k;

#pragma omp parallel private(i,j,k)
  {
#pragma omp for
  for (i=0;i<n;i++)
    for (j=0;j<n;j++)
      {
	a[i][j]=2;
	b[i][j]=3;
  	c[i][j]=0;
      }

#pragma omp for
  for (i=0;i<n;i++)
    for (j=0;j<n;j++)
      for (k=0;k<n;k++)
	c[i][j] = c[i][j] + a[i][k] * b[k][j];

#pragma omp master
  {
  for (i=0;i<n;i++)
    {
      for (j=0;j<n;j++)
	printf("%f ", c[i][j]);
      printf("\n");
    }
  }
}
  return 0;
}

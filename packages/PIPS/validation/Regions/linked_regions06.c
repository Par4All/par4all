// same as linked_regions01.c but x1 is an integer

// currently correct yet unprecise results for first loop nest.

#include <stdio.h>

int main()
{
  int N = 100;
  double A[100][100];
  for (int i=1; i<=N; i++) {
    for (int j=1; j<=N; j++) {
      int x0 = (i*j);
      int x1 = N / 2;
      if ((x0<x1)) {
        A[(N-i)-1][(i+j)-1] = 1.0;
        A[i-1][((N-i)-j)-1] = 1.0;
      }
      if ((i==j)) {
        A[i-1][j-1] = 1.0;
      }
    }
  }
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      printf("%f\n", A[i][j]);

  return 0;
}


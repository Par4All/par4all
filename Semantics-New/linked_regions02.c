// Same as linked_regions01.c, but with cleaner C code. if(1) tests
// added to obtain information about itermediate convex array regions.

#include <stdio.h>

int main()
{
  int N = 100;
  double A[100][100];
  int i, j;

  for (i=1; i<=N; i++) {
    for (j=1; j<=N; j++) {
      //int x0 = (i*j);
      //double x1 = ((double)N / 2);
      if(1) {
	if ((2*i*j<N)) {
	  if(1) {
	    A[(N-i)-1][(i+j)-1] = 1.0;
	    A[i-1][((N-i)-j)-1] = 1.0;
	  }
	}
	if ((i==j)) {
	  A[i-1][j-1] = 1.0;
	}
      }
    }
  }
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      printf("%f\n", A[i][j]);

  return 0;
}

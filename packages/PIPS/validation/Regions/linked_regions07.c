// same as linked_regions01.c but all constant integer variables
// have been replaced by their values

// There were no regions on A for the first loop nest inner block
// and consequently for the first loop nest loops.

// The result is still too complicated

#include <stdio.h>

int main()
{
  double A[100][100];
  for (int i=1; i<=100; i++) {
    for (int j=1; j<=100; j++) {
      if ((i*j)< 50) {
        A[(50-i)-1][(i+j)-1] = 1.0;
        A[i-1][((50-i)-j)-1] = 1.0;
      }
      if ((i==j)) {
        A[i-1][j-1] = 1.0;
      }
    }
  }
  for (int i=0; i<100; i++)
    for (int j=0; j<100; j++)
      printf("%f\n", A[i][j]);

  return 0;
}


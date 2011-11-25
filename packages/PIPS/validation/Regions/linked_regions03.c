// this test case is the same as linked_regions01, but the
// variables used in the main loop are declared at the
// beginning of the function

// the regions of the main loop nest are erroneously empty
// despite recent changes in normalization strategy (r20629)
// However if I revert r20629, test case linked_regions01
// yields false results.


#include <stdio.h>

int main()
{
   int ii, jj, x0;
   double x1;

   int N = 100;
   double A[100][100];

   for(ii = 1; ii <= N; ii += 1)
      for(jj = 1; jj <= N; jj += 1) {
         x0 = ii*jj;
         x1 = (double) N/2;
         if (x0<x1) {
            A[N-ii-1][ii+jj-1] = 1.0;
            A[ii-1][N-ii-jj-1] = 1.0;
         }
         if (ii==jj)
            A[ii-1][jj-1] = 1.0;
      }
   for (int i=0; i<N; i++)
     for (int j=0; j<N; j++)
       printf("%f\n", A[i][j]);
  return 0;
}

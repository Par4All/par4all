// This test case is the same as linked_regions03, but the tpips
// script sets properties to use information about the declaration and
// references to array A. The number of exceptions is reduced with
// respect to linked_region03, but there are still exceptions.

// The regions of the main loop nest is erroneously empty
// despite recent changes in normalization strategy (r20629).

// The regions of the main loop is no longer empty... but not really
// satisfying.

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

// Issue: preconditions for x0 is only obtained if transformers are
// computed in context... which makes perfect sense... but Francois
// Irigoin failed to understand it. He assumed that the precondition step
// would be enough to recover the information lost by the transformer
// computation. See comments in corresponding tpips script.
//
// Overflows occur in region computation whether conditions for x0 are
// obtained or not.

// The options TRUST_ARRAY_DECLARATIONS and TRUST_ARRAY_REFERENCES are
// not used and the regions end up outside of the array bound.

// This test case is the same as linked_regions01, but the
// variables used in the main loop are declared at the
// beginning of the function

// The regions of both loops are erroneously empty
// despite recent changes in normalization strategy (r20629)
// However if I revert r20629, test case linked_regions01
// yields false results.

// For an analysis of the convex array regions, see comments in linked_regions04.c

#include <stdio.h>

int main()
{
   int ii, jj, x0;
   double x1;

   int N = 100;
   double A[100][100];

   for(ii = 1; ii <= N; ii += 1)
     /* The region for A computed here contains the point N=100,
	ii==1, jj==1, phi1==0, phi2==0 */
      for(jj = 1; jj <= N; jj += 1) {
	if(1) {
	  x0 = ii*jj;
	  x1 = (double) N/2;
	  if (x0<x1) 
	    if(1) {
	      A[N-ii-1][ii+jj-1] = 1.0;
	      A[ii-1][N-ii-jj-1] = 1.0;
	    }
	  if (ii==jj)
            A[ii-1][jj-1] = 1.0;
	}
      }
   for (int i=0; i<N; i++)
     for (int j=0; j<N; j++)
       printf("%f\n", A[i][j]);
  return 0;
}

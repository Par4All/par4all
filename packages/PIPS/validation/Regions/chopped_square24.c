// Goal make sure that preconditions and regions stay simple when a
// simple set of constraints exists.

// This example was designed folowing the linked_region cases.

// The chopping of one corner element could occur on any of the four
// corners. Here, corner (99,99) is not removed as in chopped_square14,
// but a convex hull is preserved.

// The worry: redundancy elimination and normalization end up with
// rational constraints minimizing the number of constraints at the cost
// of coefficients with high magnitude. See corresponding TRAC ticket.

// Observation: the result is correct, the number of constraints is
// not minimized, but the simple constraint phi1+phi2<=197 ends up replaced
// by several, probably redundant as far as integer points are concerned,
// constraints with large coefficients:
//
//    99PHI2<=98PHI1+9801, - redundant: 9801 = 99*99 and phi1>=0
//    50PHI1<=49PHI2+4950, - redundant: 4950 = 50*99 and phi2>=0
//    100PHI1+99PHI2<=19701 - redundant: 19701 = 100*99+99*99

// Note also that regions are not simplified with
// sc_bounded_normalization(). This shows for A[N/2][N/2]

#include <stdio.h>

int main()
{
  int ii, jj;

  int N = 100;
  double A[100][100];

  for(ii = 0; ii < N; ii += 1)
    for(jj = 0; jj < N; jj += 1) {
      if(1) {
	A[ii][jj] = 1.0;
	A[N/2][N/2] = 0.;
      }
    }

  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      printf("%f\n", A[i][j]);

  return 0;
}

// Goal make sure that preconditions and regions stay simple when a
// simple set of constraints exists. They do not here: lots of
// arithmetic errors occur, even in this very simple case.

// This example was designed folowing the linked_region cases.

// The chopping of one corner element could occur on any of the four
// corners. Here, corner (0,0) is removed as in chopped_square01,
// but a convex hull with an internal point (!) is added.

// The worry: redundancy elimination and normalization end up with
// rational constraints minimizing the number of constraints at the cost
// of coefficients with high magnitude. See corresponding TRAC ticket.

// Observation: the result is correct, the number of constraints is
// not minimized, but the simple constraint phi1+phi2>=0 ends up replaced
// by two, probably redundant as far as integer points are concerned,
// constraints with large coefficients:
//
// 9801<=29302PHI1+9801PHI2, 100<=100PHI1+297PHI2

#include <stdio.h>

int main()
{
  int ii, jj;

  int N = 100;
  double A[100][100];

  for(ii = 0; ii < N; ii += 1)
    for(jj = 0; jj < N; jj += 1) {
      if(1) {
	if(ii>0 || jj>0) {
	  A[ii][jj] = 1.0;
	}
	A[N/2][N/2] = 0.;
      }
    }

  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      printf("%f\n", A[i][j]);

  return 0;
}

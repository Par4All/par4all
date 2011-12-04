// Goal make sure that preconditions and regions stay simple when a
// simple set of constraints exists.

// This example was designed folowing the linked_region cases.

// The chopping of one corner element could occur on any of the four
// corners. Here, corner (0,99) is removed.

// The worry: redundancy elimination and normalization end up with
// rational constraints minimizing the number of constraints at the cost
// of coefficients with high magnitude. See corresponding TRAC ticket.

#include <stdio.h>

int main()
{
  int ii, jj;

  int N = 100;
  double A[100][100];

  for(ii = 0; ii < N; ii += 1)
    for(jj = 0; jj < N; jj += 1) {
      if(ii>0 || jj<N-1) {
	A[ii][jj] = 1.0;
      }
    }

  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      printf("%f\n", A[i][j]);

  return 0;
}

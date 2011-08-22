/* Check that A[i] is scalarized in the two successive loop nests

   NOTE: derived from scalarization17
 */


#include <stdio.h>

int SIZE = 10;

void scalarization18(double A[SIZE], double B[SIZE][SIZE])
{
  int i,j;

  for(i=0 ; i < SIZE ; i++)
    for(j=0 ; j < SIZE ; j++)
      A[i] = B[j][i] + A[i];

  for(i=0 ; i < SIZE ; i++)
    for(j=0 ; j < SIZE ; j++)
      A[i] = B[j][i] + A[i];
}

main()
{
  double A[SIZE], B[SIZE][SIZE];
  int i;
  for(i=0 ; i < SIZE ; i++)
    A[i] = 0.;
  scalarization18(A, B);
  printf ("%f\n", A[0]);
}


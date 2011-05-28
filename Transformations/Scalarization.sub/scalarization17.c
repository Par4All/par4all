/* Expected result:
   A[i] is scalarized: the scalar is copied in from A[i],
   and then back into A[i]
   
   NOTE: derived from scalarization09, with additional context
   in order to obtain effective IN and OUT regions.
 */


#include <stdio.h>

int SIZE = 10;
    
void scalarization17(double A[SIZE], double B[SIZE][SIZE])
{
  int i,j;
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
  scalarization17(A, B);
  printf ("%f\n", A[0]);
}


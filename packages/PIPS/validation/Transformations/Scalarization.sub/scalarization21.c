/* Check that we don't forget to scalarize: B[i][j] is referenced several times
 */


#include <stdio.h>

int SIZE = 10;
    
void scalarization21(double A[SIZE][SIZE], double B[SIZE][SIZE])
{
  int i,j;
  for(i=0 ; i < SIZE ; i++)
    for(j=0 ; j < SIZE ; j++) {
      A[i][j] = B[i][j] * B[i][j] + B[i][j];
    }
}

main()
{
  double A[SIZE][SIZE], B[SIZE][SIZE];
  int i;

  scalarization21(A, B);

  printf ("%f\n", A[0][0]);
}


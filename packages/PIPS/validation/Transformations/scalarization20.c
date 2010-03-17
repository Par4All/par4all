/* Check that we don't forget to scalarize: B[i][j] is used several times
 */


#include <stdio.h>

int SIZE = 10;
    
void scalarization20(double B[SIZE][SIZE])
{
  int i,j;
  for(i=0 ; i < SIZE ; i++)
    for(j=0 ; j < SIZE ; j++) {
      B[i][j] = 0.;
      B[i][j] = B[i][j] + B[i][j];
      B[i][j] = B[i][j] * B[i][j];
    }
}

main()
{
  double B[SIZE][SIZE];
  int i;

  scalarization20(B);

  printf ("%f\n", B[0][0]);
}


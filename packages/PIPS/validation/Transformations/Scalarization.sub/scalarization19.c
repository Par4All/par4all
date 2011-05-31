/* Check that we don't scalarize needlessly: B[i][j] is used only once.
 */


#include <stdio.h>

int SIZE = 10;
    
void scalarization19(double B[SIZE][SIZE])
{
  int i,j;
  for(i=0 ; i < SIZE ; i++)
    for(j=0 ; j < SIZE ; j++)
      B[i][j] = 0.;
}

main()
{
  double B[SIZE][SIZE];
  int i;

  scalarization19(B);

  printf ("%f\n", B[0][0]);
}


/* Expected result:
   A[i] can be scalarized, but the scalar must be copied in from A[I],
   and then back into A[i]

   NOTE: no scalarization (LD, 19 May 2009)
 */

int SIZE = 10;
    
void scalarization09(double A[SIZE], double B[SIZE][SIZE])
{
  int i,j;
  for(i=0 ; i < SIZE ; i++)
    for(j=0 ; j < SIZE ; j++)
      A[i] = B[j][i] + A[i];
}



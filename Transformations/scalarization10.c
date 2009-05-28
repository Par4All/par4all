/* Expected result:
   A[i] can be scalarized, but the scalar must be copied in from A[I],
   and then back into A[i]

   NOTE: no scalarization (LD, 19 May 2009)
 */

int SIZE = 10;
    
double get(double f[SIZE],int i) {
  return f[i];
}

void scalarization10(double A[SIZE], double B[SIZE][SIZE])
{
  int i,j;
  for(i=0 ; i < SIZE ; i++)
    for(j=0 ; j < SIZE ; j++)
      A[i] = B[j][i] + get(A, i);
}



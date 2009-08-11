/* Expected result: A[i] can be scalarized, the scalar must be copied
   in from A[I], but not back into A[i], because it's never used in
   this context: the scalarization10 is never called.

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



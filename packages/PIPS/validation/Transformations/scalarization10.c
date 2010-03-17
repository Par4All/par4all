/* Expected result: A[i] is not scalarized, because:

   - A[i] is referenced many times inside a j-loop, so we would like
     to scalarize it;

   - BUT there is a dependence cycle between get(A,i) and A[i], so the
     scalarized version should include a copy-back from the scalar to
     A[i] at each j iteration.

   Conclusion: this copy back destroys the expected profit. Anyway, we
   don't take any risk regarding hidden references such as get(A, i),
   see "Legality Test" in scalarization.c.

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

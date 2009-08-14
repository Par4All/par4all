/* Expected result: A[i] must not be scalarized, because it's not
   profitable, and potentially not legal.
   
   NOTE: this test is derived from scalarization10.

 */

int SIZE = 10;
    
double get(double f[SIZE],int i) {
  return f[i];
}

void scalarization22(double A[SIZE], double B[SIZE])
{
  int i,j;
  for(i=0 ; i < SIZE ; i++)
    A[i] = B[i] + get(A, i);
}

/* Expected result: A[i] could be scalarized, as it is profitable due
   to the many references to A[i]. However, scalarizing A[i] may break
   dependence arcs because of hidden reference get(A, i).
   
   NOTE: this test is derived from scalarization22, with more
   references to A[i].

 */

int SIZE = 10;
    
double get(double f[SIZE],int i) {
  return f[i];
}

void scalarization23(double A[SIZE], double B[SIZE])
{
  int i,j;
  for(i=0 ; i < SIZE ; i++)
    A[i] = B[i] + A[i]*A[i] + get(A, i);
}

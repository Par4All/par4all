/* Expected result: A[i] could be scalarized, as it is profitable here
   due to the many references to A[i].

   NOTES:

   This test is derived from scalarization24, by separating the
   side-effects and the hidden reference to A[i] into two different
   statements.

   Conclusion: no scalarization, see NOTES in test case
   scalarization24.
*/

int SIZE = 10;

double get(double f[SIZE],int i) {
  return f[i];
}

void scalarization25(double A[SIZE], double B[SIZE])
{
  int i,j;
  for(i=0 ; i < SIZE ; i++) {
    A[i] = B[i] + A[i] * (++A[i]);
    A[i] = A[i] + get(A, i);
  }
}

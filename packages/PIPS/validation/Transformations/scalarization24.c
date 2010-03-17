/* Expected result: A[i] could be scalarized, as it might be
   profitable here due to the many references to A[i].
   
   NOTES: 

   This test is derived from scalarization23, with side-effects on A[i].

   The C standard discourages the use of such side-effects, by not
   specifying the result of the evaluation.

   To get a correct result, the expression should be converted to
   three-address expressions and an update of A[i] be added.

   Conclusion: programming effort to handle such cases in PIPS seems
   to be much greated than the expected benefits on non-contrived
   codes. No scalarization here.


*/

int SIZE = 10;
    
double get(double f[SIZE],int i) {
  return f[i];
}

void scalarization24(double A[SIZE], double B[SIZE])
{
  int i,j;
  for(i=0 ; i < SIZE ; i++)
    A[i] = B[i] + A[i] * (++A[i]) + get(A, i);
}

// Attempt at creating problems with a square region, but for point (0,10)

// FI: I do not get exactly the result I expected, but a similar
// result. The constraints PHI2<=PHI1+9 and 0<=PHI1 are transformed into
// PHI2<=10PHI1+9, 0<=9PHI1+10PHI2. The first new constraint eliminates
// point (0,10), using (-1,-1) and (1,19). I do not understand the second one.

#include <stdio.h>

int main()
{
  int i;

   int N = 11;
   double A[N][N];

   for(i = 0; i < N-1; i++) {
     A[0][i] = 1.0;
     A[i+1][10] = 1.0;
     A[10][0] = 1.0;
   }

   for (int i=0; i<N; i++)
     for (int j=0; j<N; j++)
       printf("%f\n", A[i][j]);

  return 0;
}

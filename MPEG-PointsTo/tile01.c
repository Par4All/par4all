// The MIN function is not defined. PIPS identifies it with the Fortran
// MIN intrinsics

extern int N;
void test(int n, int m, int a[n][m])
{
   int i, j;
   //PIPS generated variable
   int I_0;
   {
      //PIPS generated variable
      int I_3 = (n-1)/N;
      for(I_0 = 0; I_0 <= I_3; I_0 += 1) {
         //PIPS generated variable
         int I_1 = MIN(n-1, I_0*N+(N-1)), I_2 = I_0*N;
         for(i = I_2; i <= I_1; i += 1)
            for(j = 0; j <= m-1; j += 1)
               a[i][j] = 0;
      }
   }
}

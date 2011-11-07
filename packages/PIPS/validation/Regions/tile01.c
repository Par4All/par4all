// Test if the region of a tile is precise
void tile01(int n, int m, int a[n][m])
{
   int i, j;
   int N;
   int I_0;
   // FI : it the outer loops had the tile sizes as loop 
   // increment it would help
   for(I_0 = 1; I_0 <= (n-1)/N+1; I_0 += 1) {
      for(i = 1; i <= MIN(n-1, (I_0-1)*N+(N-1))-(I_0-1)*N+1; i += 1)
         for(j = 1; j <= m; j += 1)
            a[(I_0-1)*N+(i-1)][j-1] = 0;
      i = MAX(MIN(n-1, (I_0-1)*N+(N-1))-(I_0-1)*N+1, 0)+(I_0-1)*N;
   }
   I_0 = MAX((n-1)/N+1, 0);
}


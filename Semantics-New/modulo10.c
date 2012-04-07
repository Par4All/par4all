#include <stdio.h>

void kernel(unsigned int ni, unsigned int nj, unsigned int nk, unsigned int nl, float alpha, float beta, float A[ni][nk], float B[nk][nj], float D[ni][nl])
{
  //PIPS generated variable
  int i, j;
  register float D_0;
  for(i = 0; i < ni; i += 1)
    for(j = 0; j < nj; j += 1) {
      int k;
      if (i<=ni-1&&j<=nj-1) {
        D_0 = 0;
unroll:
        for(k = 0; k < nk; k += 1)
          D_0 += alpha*A[i][k]*B[k][j];
    }
    D[i][j] = D_0;
  }
}



void init_array(unsigned int ni, unsigned int nj, unsigned int nk, unsigned int nl,
		float *alpha,
		float *beta,
		float A[ni][nk],
		float B[nk][nj],
		float D[ni][nl])
{
  int i, j;

  *alpha = 32412;
  *beta = 2123;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = ((float) i*j) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = ((float) i*(j+1)) / nj;
}


int main(int argc, char **argv)
{
   int ni = 1024;
   int nj = 1024;
   int nk = 1024;
   int nl = 1024;

   float alpha;
   float beta;
   float A[ni][nk];
   float B[nk][nj];
   float D[ni][nl];
   
   /* Initialize array(s). */
   
   
   
   init_array(ni, nj, nk, nl, &alpha, &beta, A, B, D);
   
   
   /* Run kernel. */
   
   
   
   
   kernel(ni, nj, nk, nl, alpha, beta, A, B, D);
   {
      int i;

      for(i = 0; i <= 9; i += 1)
         printf("%f\n", D[i][i]);
   }
   return 0;
}

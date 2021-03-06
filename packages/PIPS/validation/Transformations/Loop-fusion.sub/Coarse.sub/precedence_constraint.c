#include <stdio.h>

int main(int argc, char *argv[])
{
   int A[360];
   int C[360];
   int E[360];
   int j, l, m;
   C[0] = 1;


   // This outer loop can be fused with the third one
   // The reordering of statements after fusion used to 
   // trigger a bug that then allowed to fuse the last loop
   for(j = 0; j <= 359; j += 1)
     A[j] = 1;

   // Nothing to fuse in this loop nest
   for(l = 0; l <= 359; l += 1)
     C[l] = 2;

   // The outermost loop can be fused with the first loop nest  
   for(m = 0; m <= 359; m += 1)
     E[m] += A[m]*C[m+1];


   printf("Result should be 2, wrong fusion produce 1 : '%d'\n",E[0]);
   return 0;
}


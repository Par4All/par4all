// Example from Ticket 531

// Required to generate an OUT region without a main()
#include <stdio.h>

// This was an excerpt that must be changed to usable in validataion

//P4A_accel_kernel p4a_kernel_main_1(int i, int j, double C[2048][2048], double D[2048][2048], double E[2048][2048], int ni, int nj, int nl)

double p4a_kernel_main_1(int i, int j, double C[2048][2048], double D[2048][2048], double E[2048][2048], int ni, int nj, int nl)
{
   //PIPS generated variable
   int k;
   // Loop nest P4A end
   if (i<=ni-1&&j<=nl-1) {
      E[i][j] = 0;
      for(k = 0; k <= nj-1; k += 1)
         E[i][j] += C[i][k]*D[k][j];
   }
   printf("%g\n", E[0][0]);
   return E[0][0];
}

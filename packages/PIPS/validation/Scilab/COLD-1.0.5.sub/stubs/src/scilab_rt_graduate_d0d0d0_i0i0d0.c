
#include <stdio.h>

#include <stdlib.h>

void scilab_rt_graduate_d0d0d0_i0i0d0(double scalarin0, 
    double scalarin1, 
    double scalarin2,
     int* scalarout0, 
    int* scalarout1, 
    double* scalarout2)
{
  printf("%f", scalarin0);

  printf("%f", scalarin1);

  printf("%f", scalarin2);

  *scalarout0 = rand();

  *scalarout1 = rand();

  *scalarout2 = rand();

}


#include <stdio.h>

#include <stdlib.h>

void scilab_rt_graduate_d0d0d0d0_d0d0i0(double scalarin0, 
    double scalarin1, 
    double scalarin2, 
    double scalarin3,
     double* scalarout0, 
    double* scalarout1, 
    int* scalarout2)
{
  printf("%f", scalarin0);

  printf("%f", scalarin1);

  printf("%f", scalarin2);

  printf("%f", scalarin3);

  *scalarout0 = rand();

  *scalarout1 = rand();

  *scalarout2 = rand();

}

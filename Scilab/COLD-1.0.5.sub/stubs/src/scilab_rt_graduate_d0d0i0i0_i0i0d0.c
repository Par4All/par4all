
#include <stdio.h>

#include <stdlib.h>

void scilab_rt_graduate_d0d0i0i0_i0i0d0(double scalarin0, 
    double scalarin1, 
    int scalarin2, 
    int scalarin3,
     int* scalarout0, 
    int* scalarout1, 
    double* scalarout2)
{
  printf("%f", scalarin0);

  printf("%f", scalarin1);

  printf("%d", scalarin2);

  printf("%d", scalarin3);

  *scalarout0 = rand();

  *scalarout1 = rand();

  *scalarout2 = rand();

}

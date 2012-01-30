
#include <stdio.h>

#include <stdlib.h>

void scilab_rt_graduate_d0d0i0d0_i0i0i0(double scalarin0, 
    double scalarin1, 
    int scalarin2, 
    double scalarin3,
     int* scalarout0, 
    int* scalarout1, 
    int* scalarout2)
{
  printf("%f", scalarin0);

  printf("%f", scalarin1);

  printf("%d", scalarin2);

  printf("%f", scalarin3);

  *scalarout0 = rand();

  *scalarout1 = rand();

  *scalarout2 = rand();

}

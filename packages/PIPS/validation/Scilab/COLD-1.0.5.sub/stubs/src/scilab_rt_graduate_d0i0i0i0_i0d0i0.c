
#include <stdio.h>

#include <stdlib.h>

void scilab_rt_graduate_d0i0i0i0_i0d0i0(double scalarin0, 
    int scalarin1, 
    int scalarin2, 
    int scalarin3,
     int* scalarout0, 
    double* scalarout1, 
    int* scalarout2)
{
  printf("%f", scalarin0);

  printf("%d", scalarin1);

  printf("%d", scalarin2);

  printf("%d", scalarin3);

  *scalarout0 = rand();

  *scalarout1 = rand();

  *scalarout2 = rand();

}

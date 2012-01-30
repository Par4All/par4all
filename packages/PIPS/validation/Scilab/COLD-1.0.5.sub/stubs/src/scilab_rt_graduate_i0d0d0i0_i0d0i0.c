
#include <stdio.h>

#include <stdlib.h>

void scilab_rt_graduate_i0d0d0i0_i0d0i0(int scalarin0, 
    double scalarin1, 
    double scalarin2, 
    int scalarin3,
     int* scalarout0, 
    double* scalarout1, 
    int* scalarout2)
{
  printf("%d", scalarin0);

  printf("%f", scalarin1);

  printf("%f", scalarin2);

  printf("%d", scalarin3);

  *scalarout0 = rand();

  *scalarout1 = rand();

  *scalarout2 = rand();

}

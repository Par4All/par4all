
#include <stdio.h>

#include <stdlib.h>

void scilab_rt_graduate_i0d0d0d0_d0i0i0(int scalarin0, 
    double scalarin1, 
    double scalarin2, 
    double scalarin3,
     double* scalarout0, 
    int* scalarout1, 
    int* scalarout2)
{
  printf("%d", scalarin0);

  printf("%f", scalarin1);

  printf("%f", scalarin2);

  printf("%f", scalarin3);

  *scalarout0 = rand();

  *scalarout1 = rand();

  *scalarout2 = rand();

}

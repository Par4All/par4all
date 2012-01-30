
#include <stdio.h>

#include <stdlib.h>

void scilab_rt_graduate_i0i0d0_i0d0d0(int scalarin0, 
    int scalarin1, 
    double scalarin2,
     int* scalarout0, 
    double* scalarout1, 
    double* scalarout2)
{
  printf("%d", scalarin0);

  printf("%d", scalarin1);

  printf("%f", scalarin2);

  *scalarout0 = rand();

  *scalarout1 = rand();

  *scalarout2 = rand();

}

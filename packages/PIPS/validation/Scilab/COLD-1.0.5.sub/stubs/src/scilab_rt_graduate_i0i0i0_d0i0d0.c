
#include <stdio.h>

#include <stdlib.h>

void scilab_rt_graduate_i0i0i0_d0i0d0(int scalarin0, 
    int scalarin1, 
    int scalarin2,
     double* scalarout0, 
    int* scalarout1, 
    double* scalarout2)
{
  printf("%d", scalarin0);

  printf("%d", scalarin1);

  printf("%d", scalarin2);

  *scalarout0 = rand();

  *scalarout1 = rand();

  *scalarout2 = rand();

}

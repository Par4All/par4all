
#include <stdio.h>

#include <stdlib.h>

void scilab_rt_graduate_i0i0_i0i0i0(int scalarin0, 
    int scalarin1,
     int* scalarout0, 
    int* scalarout1, 
    int* scalarout2)
{
  printf("%d", scalarin0);

  printf("%d", scalarin1);

  *scalarout0 = rand();

  *scalarout1 = rand();

  *scalarout2 = rand();

}

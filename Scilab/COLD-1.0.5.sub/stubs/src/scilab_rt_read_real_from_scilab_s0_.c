
#include <stdio.h>

double scilab_rt_read_real_from_scilab_s0_(char* in0)
{
  float val;

  printf("%s", in0);
  scanf("%f", &val);
  return val;
}


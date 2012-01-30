
#include <stdio.h>

int scilab_rt_read_int_from_scilab_s0_(char* in0)
{
  int val;

  printf("%s", in0);
  scanf("%d", &val);
  return val;
}


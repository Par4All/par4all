
#include <stdio.h>

double scilab_rt_mfscanf_i0i0s0_(int iter, int fd, char* format)
{
  printf("%d",iter);
  printf("%d",fd);
  printf("%s",format);

  return iter;
}


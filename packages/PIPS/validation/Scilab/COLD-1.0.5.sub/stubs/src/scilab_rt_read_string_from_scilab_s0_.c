
#include <stdio.h>
#include <stdlib.h>

char* scilab_rt_read_string_from_scilab_s0_(char* in0)
{
  char* s = (char*) malloc(128);

  printf("%s", in0);
  scanf("%s", s);
  return s;
}


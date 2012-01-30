
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

char* scilab_rt_ascii_i0_(int res)
{
  char* s = (char*)malloc(2*sizeof(char));
  s[0] = (char) res;;
  s[1] = 0;
  return s;
}


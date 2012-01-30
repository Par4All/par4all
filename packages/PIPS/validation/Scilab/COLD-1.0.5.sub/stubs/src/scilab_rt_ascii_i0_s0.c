
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void scilab_rt_ascii_i0_s0(int res, char** s)
{
  *s = (char*)malloc(2*sizeof(char));
  (*s)[0] = (char) res;;
  (*s)[1] = 0;
}


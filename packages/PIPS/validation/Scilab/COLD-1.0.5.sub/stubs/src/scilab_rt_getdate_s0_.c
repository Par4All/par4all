
#include <stdio.h>

double scilab_rt_getdate_s0_(char* in0) 
{
  float val0 = 0;;

  if(*in0 != 's') {
    scanf("%f", &val0);
  }
  return val0;

}


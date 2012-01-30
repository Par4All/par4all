
#include <stdio.h>
#include <complex.h>

double complex scilab_rt_read_complex_from_scilab_s0_(char* in0)
{
  float val;

  printf("%s", in0);
  scanf("%f", &val);
  return val + val*I;
}


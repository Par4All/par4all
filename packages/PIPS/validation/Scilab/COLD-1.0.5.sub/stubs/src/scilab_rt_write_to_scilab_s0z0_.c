
#include <stdio.h>
#include <complex.h>

void scilab_rt_write_to_scilab_s0z0_(char* s, double complex x)
{
  printf("%s",s);
  printf("%f",creal(x));
  printf("%f",cimag(x));
}



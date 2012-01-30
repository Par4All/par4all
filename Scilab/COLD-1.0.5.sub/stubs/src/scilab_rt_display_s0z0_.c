
#include <stdio.h>
#include <complex.h>

void scilab_rt_display_s0z0_(char* name, double complex in0)
{
  printf("%s %g %g\n", name, creal(in0), cimag(in0));
}


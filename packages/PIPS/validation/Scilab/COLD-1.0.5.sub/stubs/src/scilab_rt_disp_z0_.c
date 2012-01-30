
#include <stdio.h>
#include <complex.h>

void scilab_rt_disp_z0_(double complex in0)
{
  printf("%g %g\n", creal(in0), cimag(in0));
}


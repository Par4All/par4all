
#include <stdio.h>
#include <complex.h>


void scilab_rt_read_from_scilab_s0_z0_(char* s, double complex* x)
{
  printf("%s",s);
  *x = 1 + 1*I;
  printf("%f %f",creal(*x), cimag(*x));
}



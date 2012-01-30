
#include <stdio.h>
#include <complex.h>

void scilab_rt_display_s0z2_(char* name, int si00, int si01, double complex in0[si00][si01])
{
  int i;
  int j;

  double val0Real = 0;
  double val0Imag = 0;
	
  printf("%s\n", name);

  for (i = 0; i < si00; ++i) {
    for (j = 0; j < si01; ++j) {
      val0Real += creal(in0[i][j]);
      val0Imag += cimag(in0[i][j]);
    }
  }
  printf("%g %g\n", val0Real, val0Imag);

}


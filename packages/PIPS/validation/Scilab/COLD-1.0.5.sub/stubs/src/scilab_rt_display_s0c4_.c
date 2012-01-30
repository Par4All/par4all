
#include <stdio.h>
#include <complex.h>

void scilab_rt_display_s0c4_(char* name, int si00, int si01, int si02, int si03, double complex in0[si00][si01][si02][si03])
{
  int i;
  int j;
  int k;
  int l;

  double val0Real = 0;
  double val0Imag = 0;
	
  printf("%s\n", name);

  for (i = 0; i < si00; ++i) {
    for (j = 0; j < si01; ++j) {
      for (k = 0; k < si02; ++k) {
        for (l = 0; l < si03; ++l) {
          val0Real += creal(in0[i][j][k][l]);
          val0Imag += cimag(in0[i][j][k][l]);
        }
      }
    }
  }
  printf("%g %g\n", val0Real, val0Imag);
}




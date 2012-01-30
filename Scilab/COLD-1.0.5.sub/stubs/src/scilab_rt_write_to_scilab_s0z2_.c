
#include <stdio.h>
#include <complex.h>

void scilab_rt_write_to_scilab_s0z2_(char* s, int xsize, int ysize, double complex aMatrix[xsize][ysize])
{

  int i;
  int j;

  double val1R = 0;
  double val1I = 0;

  printf("%s",s);
  for (i = 0; i < xsize; ++i) {
    for (j = 0; j < ysize; ++j) {
      val1R += creal(aMatrix[i][j]);
      val1I += cimag(aMatrix[i][j]);
    }
  }

  printf("%f",val1R);
  printf("%f",val1I);

}



#include <stdio.h>

void scilab_rt_write_to_scilab_s0d2_(char* s, int xsize, int ysize, double aMatrix[xsize][ysize])
{

  int i;
  int j;
  double val1 = 0;

  printf("%s",s);

  for (i = 0; i < xsize; ++i) {
    for (j = 0; j < ysize; ++j) {
      val1 += aMatrix[i][j];
    }
  }

  printf("%f",val1);

}


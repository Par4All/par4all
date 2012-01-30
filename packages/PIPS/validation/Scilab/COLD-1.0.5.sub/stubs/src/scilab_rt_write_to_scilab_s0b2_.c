
#include <stdio.h>

void scilab_rt_write_to_scilab_s0b2_(char* s, int xsize, int ysize, int aMatrix[xsize][ysize])
{

  int i;
  int j;
  int val1 = 0;

  printf("%s",s);

  for (i = 0; i < xsize; ++i) {
    for (j = 0; j < ysize; ++j) {
      val1 += aMatrix[i][j];
    }
  }

  printf("%d",val1);

}


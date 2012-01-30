
#include <stdio.h>

void scilab_rt_write_to_scilab_s0s2_(char* s, int xsize, int ysize, char* aMatrix[xsize][ysize])
{

  int i;
  int j;

  printf("%s",s);

  for (i = 0; i < xsize; ++i) {
    for (j = 0; j < ysize; ++j) {
      printf("%s",aMatrix[i][j]);
    }
  }


}


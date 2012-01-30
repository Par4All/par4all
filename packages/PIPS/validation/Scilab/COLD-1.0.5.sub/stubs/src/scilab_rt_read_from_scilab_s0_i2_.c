
#include <stdio.h>

void scilab_rt_read_from_scilab_s0_i2(char* s, int nx, int ny, int aMatrix[nx][ny])
{
  int i,j;
  int val=0;

  printf("%s", s);

  scanf("%d", &val);
  for (i = 0; i < nx; ++i) {
    for (j = 0; j < ny; ++j) {
      aMatrix[i][j] = val;
    }
  }


}


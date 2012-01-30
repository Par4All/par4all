
#include <stdio.h>

void scilab_rt_read_from_scilab_s0_d2(char* s, int nx, int ny, double aMatrix[nx][ny])
{
  int i,j;
  float val=0;

  printf("%s", s);

  scanf("%f", &val);
  for (i = 0; i < nx; ++i) {
    for (j = 0; j < ny; ++j) {
      aMatrix[i][j] = val;
    }
  }

}


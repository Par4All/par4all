
#include <stdio.h>
#include <complex.h>

void scilab_rt_read_from_scilab_s0_z2(char* s, int nx, int ny, double complex aMatrix[nx][ny])
{
  int i,j;
  float val;

  printf("%s", s);

  scanf("%f", &val);
  for (i = 0; i < nx; ++i) {
    for (j = 0; j < ny; ++j) {
      aMatrix[i][j] = val + val*I;
    }
  }

}


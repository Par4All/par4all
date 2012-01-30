
#include <stdio.h>
#include <stdlib.h>

void scilab_rt_read_from_scilab_s0_s2(char* s, int nx, int ny, char* aMatrix[nx][ny])
{
  int i,j;
  char* val = (char*)malloc(128);

  printf("%s", s);

  scanf("%s", val);
  for (i = 0; i < nx; ++i) {
    for (j = 0; j < ny; ++j) {
      aMatrix[i][j] = val;
    }
  }



}


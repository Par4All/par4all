
#include <stdio.h>

void scilab_rt_disp_i2_(int si00, int si01, int in0[si00][si01])
{
  int i;
  int j;

  int val0 = 0;
  for (i = 0; i < si00; ++i) {
    for (j = 0; j < si01; ++j) {
      val0 += in0[i][j];
    }
  }
  printf("%d\n", val0);
}


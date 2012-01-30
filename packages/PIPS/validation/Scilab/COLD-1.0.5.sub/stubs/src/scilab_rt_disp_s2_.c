#include <stdio.h>

void scilab_rt_disp_s2_(int si00, int si01, char* in0[si00][si01])
{
  int i;
  int j;

  for (i = 0; i < si00; ++i) {
    for (j = 0; j < si01; ++j) {
      printf("%s", in0[i][j]);
    }
  }

}



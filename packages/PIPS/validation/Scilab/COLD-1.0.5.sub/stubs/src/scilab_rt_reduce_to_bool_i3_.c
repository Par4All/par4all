
#include <stdio.h>

int scilab_rt_reduce_to_bool_i3_(int si00, int si01, int si02, int in0[si00][si01][si02])
{

  int i;
  int j;
  int k;

  int val1 = 0;
  for (i = 0; i < si00; ++i) {
    for (j = 0; j < si01; ++j) {
      for (k = 0; k < si02; ++k) {
        val1 += in0[i][j][k];
      }
    }
  }
  return val1;
}


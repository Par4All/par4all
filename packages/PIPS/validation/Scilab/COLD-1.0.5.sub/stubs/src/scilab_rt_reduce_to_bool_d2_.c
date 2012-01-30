
#include <stdio.h>

int scilab_rt_reduce_to_bool_d2_(int si00, int si01, double in0[si00][si01])
{

  int i;
  int j;

  double val1 = 0;
  for (i = 0; i < si00; ++i) {
    for (j = 0; j < si01; ++j) {
      val1 += in0[i][j];
    }
  }
  return (int) val1;

}


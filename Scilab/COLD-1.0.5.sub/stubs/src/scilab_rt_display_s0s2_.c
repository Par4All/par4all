#include <stdio.h>

void scilab_rt_display_s0s2_(char* name, int si00, int si01, char* in0[si00][si01])
{
  int i;
  int j;

  printf("%s\n", name);

  for (i = 0; i < si00; ++i) {
    for (j = 0; j < si01; ++j) {
      printf("%s", in0[i][j]);
    }
  }

}



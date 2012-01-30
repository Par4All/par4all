
#include <stdio.h>

void scilab_rt_disp_s3_(int si00, int si01, int si02, char* in0[si00][si01][si02])
{
  int i;
  int j;
  int k;

  for (i = 0; i < si00; ++i) {
    for (j = 0; j < si01; ++j) {
      for (k = 0; k < si02; ++k) {
        printf("%s", in0[i][j][k]);
      }
    }
  }

}


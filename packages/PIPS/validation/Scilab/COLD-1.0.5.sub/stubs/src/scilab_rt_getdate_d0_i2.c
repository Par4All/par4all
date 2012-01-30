
#include <stdio.h>

void scilab_rt_getdate_d0_i2(double in0, int sout00, int sout01, int out0[sout00][sout01])
{
  int i,j;
  int val0;
  scanf("%d", &val0);

  if (in0 != 0) {
    for (i = 0; i < sout00; ++i) {
      for (j = 0 ; j < sout01; ++j) {
        out0[i][j] = val0;
      }
    }
  }
}

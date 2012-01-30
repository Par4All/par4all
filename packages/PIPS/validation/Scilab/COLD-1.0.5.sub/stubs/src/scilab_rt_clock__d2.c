
#include <stdio.h>
#include <stdlib.h>

void scilab_rt_clock__d2(int sout00, int sout01, double out0[sout00][sout01])
{

  int i, j;
  float val = 0.1;

  scanf("%f", &val);
  for (i = 0; i < sout00; ++i) {
    for (j = 0; j < sout01; ++j) {
      out0[i][j] = val;
    }
  }

}


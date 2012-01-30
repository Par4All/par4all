
#include <stdio.h>

#include <stdlib.h>

void scilab_rt_qr_d2_d2d2(int sin00, int sin01, double in0[sin00][sin01],
    int sout00, int sout01, double out0[sout00][sout01],
    int sout10, int sout11, double out1[sout10][sout11])
{
  int i;
  int j;

  double val0 = 0;
  for (i = 0; i < sin00; ++i) {
    for (j = 0; j < sin01; ++j) {
      val0 += in0[i][j];
    }
  }

  for (i = 0; i < sout00; ++i) {
    for (j = 0; j < sout01; ++j) {
        out0[i][j] = val0;
    }
  }
  for (i = 0; i < sout10; ++i) {
    for (j = 0; j < sout11; ++j) {
        out1[i][j] = val0;
    }
  }
}

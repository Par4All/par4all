
#include <stdlib.h>

void scilab_rt_squeeze_s3_s3(int sin00, int sin01, int sin02, char* in0[sin00][sin01][sin02],
    int sout00, int sout01, int sout02, char* out0[sout00][sout01][sout02])
{
  int i;
  int j;
  int k;

  char* val0 = (char*) malloc(128);
  for (i = 0; i < sin00; ++i) {
    for (j = 0; j < sin01; ++j) {
      for (k = 0; k < sin02; ++k) {
        *val0 += *in0[i][j][k];
      }
    }
  }

  for (i = 0; i < sout00; ++i) {
    for (j = 0; j < sout01; ++j) {
      for (k = 0; k < sout02; ++k) {
        out0[i][j][k] = val0;
      }
    }
  }

}


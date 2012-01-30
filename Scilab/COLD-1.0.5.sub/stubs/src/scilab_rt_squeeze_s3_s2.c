
#include <stdlib.h>

void scilab_rt_squeeze_s3_s2(int in00, int in01, int in02, char* matrixin0[in00][in01][in02],
     int out00, int out01, char* matrixout0[out00][out01])
{
  int i;
  int j;
  int k;

  char* val = (char*) malloc(128);
  for (i = 0; i < in00; ++i) {
    for (j = 0; j < in01; ++j) {
      for (k = 0; k < in02; ++k) {
        *val += *matrixin0[i][j][k];
      }
    }
  }

  for (i = 0; i < out00; ++i) {
    for (j = 0; j < out01; ++j) {
        matrixout0[i][j] = val;
    }
  }
}

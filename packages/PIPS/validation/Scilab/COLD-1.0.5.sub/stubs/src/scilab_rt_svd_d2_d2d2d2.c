
#include <stdio.h>

#include <stdlib.h>

void scilab_rt_svd_d2_d2d2d2(int in00, int in01, double matrixin0[in00][in01],
     int out00, int out01, double matrixout0[out00][out01], 
    int out10, int out11, double matrixout1[out10][out11], 
    int out20, int out21, double matrixout2[out20][out21])
{
  int i;
  int j;

  double val0 = 0;
  for (i = 0; i < in00; ++i) {
    for (j = 0; j < in01; ++j) {
      val0 += matrixin0[i][j];
    }
  }

  for (i = 0; i < out00; ++i) {
    for (j = 0; j < out01; ++j) {
        matrixout0[i][j] = val0;
    }
  }
  for (i = 0; i < out10; ++i) {
    for (j = 0; j < out11; ++j) {
        matrixout1[i][j] = val0;
    }
  }
  for (i = 0; i < out20; ++i) {
    for (j = 0; j < out21; ++j) {
        matrixout2[i][j] = val0;
    }
  }
}


#include <stdio.h>

void scilab_rt_xnumb_i2i2d2d0i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    double scalarin0, 
    int in30, int in31, int matrixin3[in30][in31])
{
  int i;
  int j;

  int val0 = 0;
  int val1 = 0;
  double val2 = 0;
  int val3 = 0;
  for (i = 0; i < in00; ++i) {
    for (j = 0; j < in01; ++j) {
      val0 += matrixin0[i][j];
    }
  }
  printf("%d", val0);

  for (i = 0; i < in10; ++i) {
    for (j = 0; j < in11; ++j) {
      val1 += matrixin1[i][j];
    }
  }
  printf("%d", val1);

  for (i = 0; i < in20; ++i) {
    for (j = 0; j < in21; ++j) {
      val2 += matrixin2[i][j];
    }
  }
  printf("%f", val2);

  printf("%f", scalarin0);

  for (i = 0; i < in30; ++i) {
    for (j = 0; j < in31; ++j) {
      val3 += matrixin3[i][j];
    }
  }
  printf("%d", val3);

}

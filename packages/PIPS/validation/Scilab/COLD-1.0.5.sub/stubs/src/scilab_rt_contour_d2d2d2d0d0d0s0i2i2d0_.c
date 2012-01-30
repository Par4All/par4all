
#include <stdio.h>

void scilab_rt_contour_d2d2d2d0d0d0s0i2i2d0_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int in20, int in21, double matrixin2[in20][in21], 
    double scalarin0, 
    double scalarin1, 
    double scalarin2, 
    char* scalarin3, 
    int in30, int in31, int matrixin3[in30][in31], 
    int in40, int in41, int matrixin4[in40][in41], 
    double scalarin4)
{
  int i;
  int j;

  double val0 = 0;
  double val1 = 0;
  double val2 = 0;
  int val3 = 0;
  int val4 = 0;
  for (i = 0; i < in00; ++i) {
    for (j = 0; j < in01; ++j) {
      val0 += matrixin0[i][j];
    }
  }
  printf("%f", val0);

  for (i = 0; i < in10; ++i) {
    for (j = 0; j < in11; ++j) {
      val1 += matrixin1[i][j];
    }
  }
  printf("%f", val1);

  for (i = 0; i < in20; ++i) {
    for (j = 0; j < in21; ++j) {
      val2 += matrixin2[i][j];
    }
  }
  printf("%f", val2);

  printf("%f", scalarin0);

  printf("%f", scalarin1);

  printf("%f", scalarin2);

  printf("%s", scalarin3);

  for (i = 0; i < in30; ++i) {
    for (j = 0; j < in31; ++j) {
      val3 += matrixin3[i][j];
    }
  }
  printf("%d", val3);

  for (i = 0; i < in40; ++i) {
    for (j = 0; j < in41; ++j) {
      val4 += matrixin4[i][j];
    }
  }
  printf("%d", val4);

  printf("%f", scalarin4);

}

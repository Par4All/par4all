
#include <stdio.h>

void scilab_rt_hist3d_d2d0i0s0d2_(int in00, int in01, double matrixin0[in00][in01], 
    double scalarin0, 
    int scalarin1, 
    char* scalarin2, 
    int in10, int in11, double matrixin1[in10][in11])
{
  int i;
  int j;

  double val0 = 0;
  double val1 = 0;
  for (i = 0; i < in00; ++i) {
    for (j = 0; j < in01; ++j) {
      val0 += matrixin0[i][j];
    }
  }
  printf("%f", val0);

  printf("%f", scalarin0);

  printf("%d", scalarin1);

  printf("%s", scalarin2);

  for (i = 0; i < in10; ++i) {
    for (j = 0; j < in11; ++j) {
      val1 += matrixin1[i][j];
    }
  }
  printf("%f", val1);

}

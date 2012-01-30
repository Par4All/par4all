
#include <stdio.h>

void scilab_rt_bar_d2d2d0s0s0_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    double scalarin0, 
    char* scalarin1, 
    char* scalarin2)
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

  for (i = 0; i < in10; ++i) {
    for (j = 0; j < in11; ++j) {
      val1 += matrixin1[i][j];
    }
  }
  printf("%f", val1);

  printf("%f", scalarin0);

  printf("%s", scalarin1);

  printf("%s", scalarin2);

}

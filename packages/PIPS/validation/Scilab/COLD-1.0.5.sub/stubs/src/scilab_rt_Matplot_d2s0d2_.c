
#include <stdio.h>

void scilab_rt_Matplot_d2s0d2_(int in00, int in01, double matrixin0[in00][in01], 
    char* scalarin0, 
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

  printf("%s", scalarin0);

  for (i = 0; i < in10; ++i) {
    for (j = 0; j < in11; ++j) {
      val1 += matrixin1[i][j];
    }
  }
  printf("%f", val1);

}

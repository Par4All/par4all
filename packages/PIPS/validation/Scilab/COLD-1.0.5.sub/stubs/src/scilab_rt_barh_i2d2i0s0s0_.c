
#include <stdio.h>

void scilab_rt_barh_i2d2i0s0s0_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11], 
    int scalarin0, 
    char* scalarin1, 
    char* scalarin2)
{
  int i;
  int j;

  int val0 = 0;
  double val1 = 0;
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
  printf("%f", val1);

  printf("%d", scalarin0);

  printf("%s", scalarin1);

  printf("%s", scalarin2);

}

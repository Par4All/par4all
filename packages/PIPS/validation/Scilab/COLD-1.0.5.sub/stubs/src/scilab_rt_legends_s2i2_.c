
#include <stdio.h>

void scilab_rt_legends_s2i2_(int in00, int in01, char* matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11])
{
  int i;
  int j;

  int val1 = 0;
  for (i = 0; i < in00; ++i) {
    for (j = 0; j < in01; ++j) {
      printf("%s", matrixin0[i][j]);
    }
  }

  for (i = 0; i < in10; ++i) {
    for (j = 0; j < in11; ++j) {
      val1 += matrixin1[i][j];
    }
  }
  printf("%d", val1);

}

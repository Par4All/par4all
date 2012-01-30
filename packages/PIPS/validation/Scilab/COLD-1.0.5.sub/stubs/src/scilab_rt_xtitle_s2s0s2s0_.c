
#include <stdio.h>

void scilab_rt_xtitle_s2s0s2s0_(int in00, int in01, char* matrixin0[in00][in01], 
    char* scalarin0, 
    int in10, int in11, char* matrixin1[in10][in11], 
    char* scalarin1)
{
  int i;
  int j;

  for (i = 0; i < in00; ++i) {
    for (j = 0; j < in01; ++j) {
      printf("%s", matrixin0[i][j]);
    }
  }

  printf("%s", scalarin0);

  for (i = 0; i < in10; ++i) {
    for (j = 0; j < in11; ++j) {
      printf("%s", matrixin1[i][j]);
    }
  }

  printf("%s", scalarin1);

}

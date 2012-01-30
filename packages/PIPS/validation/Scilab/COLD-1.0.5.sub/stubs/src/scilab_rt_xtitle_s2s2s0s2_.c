
#include <stdio.h>

void scilab_rt_xtitle_s2s2s0s2_(int in00, int in01, char* matrixin0[in00][in01], 
    int in10, int in11, char* matrixin1[in10][in11], 
    char* scalarin0, 
    int in20, int in21, char* matrixin2[in20][in21])
{
  int i;
  int j;

  for (i = 0; i < in00; ++i) {
    for (j = 0; j < in01; ++j) {
      printf("%s", matrixin0[i][j]);
    }
  }

  for (i = 0; i < in10; ++i) {
    for (j = 0; j < in11; ++j) {
      printf("%s", matrixin1[i][j]);
    }
  }

  printf("%s", scalarin0);

  for (i = 0; i < in20; ++i) {
    for (j = 0; j < in21; ++j) {
      printf("%s", matrixin2[i][j]);
    }
  }

}

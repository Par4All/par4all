
#include <stdio.h>

void scilab_rt_xtitle_s2s0s0_(int in00, int in01, char* matrixin0[in00][in01], 
    char* scalarin0, 
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

  printf("%s", scalarin1);

}

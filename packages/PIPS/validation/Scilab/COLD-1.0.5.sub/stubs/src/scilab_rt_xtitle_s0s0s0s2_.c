
#include <stdio.h>

void scilab_rt_xtitle_s0s0s0s2_(char* scalarin0, 
    char* scalarin1, 
    char* scalarin2, 
    int in00, int in01, char* matrixin0[in00][in01])
{
  int i;
  int j;

  printf("%s", scalarin0);

  printf("%s", scalarin1);

  printf("%s", scalarin2);

  for (i = 0; i < in00; ++i) {
    for (j = 0; j < in01; ++j) {
      printf("%s", matrixin0[i][j]);
    }
  }

}

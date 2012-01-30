
#include <stdio.h>

void scilab_rt_bar_i0i2d0s0s0_(int scalarin0, 
    int in00, int in01, int matrixin0[in00][in01], 
    double scalarin1, 
    char* scalarin2, 
    char* scalarin3)
{
  int i;
  int j;

  int val0 = 0;
  printf("%d", scalarin0);

  for (i = 0; i < in00; ++i) {
    for (j = 0; j < in01; ++j) {
      val0 += matrixin0[i][j];
    }
  }
  printf("%d", val0);

  printf("%f", scalarin1);

  printf("%s", scalarin2);

  printf("%s", scalarin3);

}

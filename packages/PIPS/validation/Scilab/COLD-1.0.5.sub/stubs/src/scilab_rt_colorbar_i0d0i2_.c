
#include <stdio.h>

void scilab_rt_colorbar_i0d0i2_(int scalarin0, 
    double scalarin1, 
    int in00, int in01, int matrixin0[in00][in01])
{
  int i;
  int j;

  int val0 = 0;
  printf("%d", scalarin0);

  printf("%f", scalarin1);

  for (i = 0; i < in00; ++i) {
    for (j = 0; j < in01; ++j) {
      val0 += matrixin0[i][j];
    }
  }
  printf("%d", val0);

}

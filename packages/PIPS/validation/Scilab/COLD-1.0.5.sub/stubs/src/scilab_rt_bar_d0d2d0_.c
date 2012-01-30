
#include <stdio.h>

void scilab_rt_bar_d0d2d0_(double scalarin0, 
    int in00, int in01, double matrixin0[in00][in01], 
    double scalarin1)
{
  int i;
  int j;

  double val0 = 0;
  printf("%f", scalarin0);

  for (i = 0; i < in00; ++i) {
    for (j = 0; j < in01; ++j) {
      val0 += matrixin0[i][j];
    }
  }
  printf("%f", val0);

  printf("%f", scalarin1);

}

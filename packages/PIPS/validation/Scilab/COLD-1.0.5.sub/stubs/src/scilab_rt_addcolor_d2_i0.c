
#include <stdio.h>

#include <stdlib.h>

void scilab_rt_addcolor_d2_i0(int in00, int in01, double matrixin0[in00][in01],
     int* scalarout0)
{
  int i;
  int j;

  double val0 = 0;
  for (i = 0; i < in00; ++i) {
    for (j = 0; j < in01; ++j) {
      val0 += matrixin0[i][j];
    }
  }
  printf("%f", val0);

  *scalarout0 = rand();

}

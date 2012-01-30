
#include <stdio.h>

void scilab_rt_barh_d2i0i0s0s0_(int in00, int in01, double matrixin0[in00][in01], 
    int scalarin0, 
    int scalarin1, 
    char* scalarin2, 
    char* scalarin3)
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

  printf("%d", scalarin0);

  printf("%d", scalarin1);

  printf("%s", scalarin2);

  printf("%s", scalarin3);

}

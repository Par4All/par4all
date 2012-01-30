
#include <stdio.h>

void scilab_rt_display_s0i4_(char* name, int si00, int si01, int si02, int si03, int in0[si00][si01][si02][si03])
{
  int i;
  int j;
  int k;
  int l;

	int val0 = 0;
	
  printf("%s\n", name);

  for (i = 0; i < si00; ++i) {
    for (j = 0; j < si01; ++j) {
      for (k = 0; k < si02; ++k) {
        for (l = 0; l < si03; ++l) {
          val0 += in0[i][j][k][l];
        }
      }
    }
  }
  printf("%d\n", val0);
}


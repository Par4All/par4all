
#include <stdio.h>
#include <stdlib.h>

void scilab_rt_read_stringM_from_scilab_s0i0i0_s2(char* in0, int in1, int in2, int sout00, int sout01, char* out0[sout00][sout01])
{
  int i, j;
  char* val = (char*) malloc(128);

  printf("%s", in0);
  printf("%d", in1);
  printf("%d", in2);

  scanf("%s", val);
  for (i = 0; i < sout00; ++i) {
    for (j = 0; j < sout01; ++j) {
      out0[i][j] = val;
    }
  }

}



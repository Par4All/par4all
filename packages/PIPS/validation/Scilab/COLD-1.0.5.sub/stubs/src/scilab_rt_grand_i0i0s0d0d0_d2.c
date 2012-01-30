#include <stdio.h>

void scilab_rt_grand_i0i0s0d0d0_d2(int in0, int in1, char* in2, double in3, double in4,
        int sout00, int sout01, double out0[sout00][sout01])
{
  int i,j;

  printf("%s", in2);
  for (i = 0 ; i < in0 ; ++i) {
    for (j = 0 ; j < in1 ; ++j) {
      out0[i][j] = in3  + in4;
    }
  }
}



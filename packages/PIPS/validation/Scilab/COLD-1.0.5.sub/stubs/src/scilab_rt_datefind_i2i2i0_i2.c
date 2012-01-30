
#include <stdio.h>

void scilab_rt_datefind_i2i2i0_i2(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11], int tol,
    int sout00, int sout01, int out0[sout00][sout01])
{


  int i,j;
  int val0, val1;
  
  if (tol != 0) {
    for (i = 0 ; i < sin00 ; ++i){
      for (j = 0 ; j < sin01 ; ++j){
        val0 += in0[i][j];
      }
    }

    for (i = 0 ; i < sin10 ; ++i){
      for (j = 0 ; j < sin11 ; ++j){
        val1 += in1[i][j];
      }
    }

    for (i = 0 ; i < sout00 ; ++i){
      for ( j = 0 ; j < sout01 ; ++j){
        out0[i][j] = val0 + val1;
      }
    }
  }

}



void scilab_rt_max_d0i2_d2(double in0,
    int sin10, int sin11, int in1[sin10][sin11],
    int sout00, int sout01, double out0[sout00][sout01])
{
  int i,j;
  double val0=0;

  if (in0) {
    for (i = 0; i < sin10; ++i) {
      for (j = 0; j < sin11; ++j) {
        val0 += in1[i][j];
      }
    }

    for (i = 0 ; i < sout00 ; ++i){
      for(j = 0 ; j < sout01 ; ++j){
        out0[i][j] = val0;;
      }
    }
  }
}



void scilab_rt_variance_i2s0_d2(int sin00, int sin01, int in0[sin00][sin01],
    char* in1,
    int sout00, int sout01, double out0[sout00][sout01])
{
  int i,j;
  int val0 = 0;

  if (in1) {
    for (j = 0 ; j < sin01 ; ++j){
      for (i = 0 ; i < sin00 ; ++i){
        val0 += in0[i][j]; 
      }
    }
  }

  for (i = 0; i < sout00; ++i){
    for (j = 0; j < sout01; ++j){
      out0[i][j] = val0;
    }
  }

}



void scilab_rt_min_d2s0_d2(int sin00, int sin01, double in0[sin00][sin01],
    char * in1,
    int sout00, int sout01, double out0[sout00][sout01])
{

  int i,j;
  double val0=0;

  if (in1) {
    for (i = 0; i < sin00; ++i) {
      for (j = 0; j < sin01; ++j) {
        val0 += in0[i][j];
      }
    }

    for (i = 0 ; i < sout00 ; ++i){
      for(j = 0 ; j < sout01 ; ++j){
        out0[i][j] = val0;
      }
    }
  }

}


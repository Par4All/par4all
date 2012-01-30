
void scilab_rt_datevec_d0_d2(double in0, int sout00, int sout01, double out0[sout00][sout01]){

  int i,j;

  for (i = 0; i < sout00; ++i) {
    for (j = 0; j < sout01; ++j) {
      out0[i][j] = in0;
    }
  }

}

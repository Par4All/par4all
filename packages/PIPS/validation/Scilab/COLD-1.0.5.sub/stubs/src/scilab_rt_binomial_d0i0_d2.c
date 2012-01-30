
void scilab_rt_binomial_d0i0_d2(double in0, int in1, int sout00, int sout01, double out0[sout00][sout01])
{
  int i;
  int j;
  for (i = 0; i < sout00; ++i) {
    for (j = 0; j < sout01; ++j) {
      out0[i][j] = i*j*in0*in1;
    }
  }

}


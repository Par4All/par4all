
void scilab_rt_weekday_d2s0_i2s2(int sin00, int sin01, double in0[sin00][sin01],
    char* in1,
    int sout00, int sout01, int out0[sout00][sout01],
    int sout10, int sout11, char* out1[sout10][sout11])
{
  int i,j;
  double val0=0;

  if (in1) {
    for (i = 0; i < sin00; ++i) {
      for (j = 0; j < sin01; ++j) {
        val0 += in0[i][j];
      }
    }
  }

  for (i = 0; i < sout00; ++i) {
    for (j = 0; j < sout01; ++j) {
      out0[i][j] = val0;
    }
  }

  for (i = 0; i < sout10; ++i) {
    for (j = 0; j < sout11; ++j) {
      *out1[i][j] = val0;
    }
  }

}


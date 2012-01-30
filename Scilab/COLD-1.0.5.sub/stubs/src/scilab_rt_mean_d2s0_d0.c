
void scilab_rt_mean_d2s0_d0(int sin00, int sin01, double in0[sin00][sin01],
    char* in1,
    double* out0)
{
  int i;
  int j;

  double val0 = 0;

  if (in1) {
    for (i = 0; i < sin00; ++i) {
      for (j = 0; j < sin01; ++j) {
        val0 += in0[i][j];
      }
    }

    *out0 = val0;
  }
}


void scilab_rt_inttrap_d2_d0(int sin00, int sin01, double in0[sin00][sin01],
    double *out0)
{
  int i;
  int j;

  double val0 = 0;
  for (i = 0; i < sin00; ++i) {
    for (j = 0; j < sin01; ++j) {
      val0 += in0[i][j];
    }
  }

  *out0 = val0;
}



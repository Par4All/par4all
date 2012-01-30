
void scilab_rt_max_d2_d0(int sin00, int sin01, double in0[sin00][sin01],
        double *out0)
{

  int i, j;
  double val1 = 0;
  for (i = 0; i < sin00; ++i) {
    for (j = 0; j < sin01; ++j) {
        val1 += in0[i][j];
    }
  }
  *out0 = val1;

}

